use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Frame, SizeHint};
use thiserror::Error;

use crate::metrics::{HttpBodyTermination, RouterMetrics};
use crate::worker_pool::RequestLease;

#[derive(Debug, Error)]
#[error("upstream response body terminated")]
pub(crate) struct RelayError;

/// Direct upstream response body whose terminal owner retains request admission.
pub(crate) struct DirectResponseBody {
    inner: Option<reqwest::Body>,
    lease: Option<RequestLease>,
    metrics: Arc<RouterMetrics>,
    terminal: bool,
}

impl DirectResponseBody {
    pub(crate) fn new(
        inner: reqwest::Body,
        lease: RequestLease,
        metrics: Arc<RouterMetrics>,
    ) -> Self {
        Self {
            inner: Some(inner),
            lease: Some(lease),
            metrics,
            terminal: false,
        }
    }

    fn terminalize(&mut self, termination: HttpBodyTermination) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        self.metrics.record_http_body_termination(termination);
        if termination == HttpBodyTermination::UpstreamError {
            self.metrics.record_relay_failure();
            if let Some(lease) = self.lease.as_ref() {
                lease.request_immediate_probe();
            }
        }
        drop(self.inner.take());
        drop(self.lease.take());
    }

    fn fail(&mut self) -> Poll<Option<Result<Frame<Bytes>, RelayError>>> {
        self.terminalize(HttpBodyTermination::UpstreamError);
        Poll::Ready(Some(Err(RelayError)))
    }
}

impl Drop for DirectResponseBody {
    fn drop(&mut self) {
        let termination = if self
            .inner
            .as_ref()
            .is_some_and(http_body::Body::is_end_stream)
        {
            HttpBodyTermination::Complete
        } else {
            HttpBodyTermination::Dropped
        };
        self.terminalize(termination);
    }
}

impl http_body::Body for DirectResponseBody {
    type Data = Bytes;
    type Error = RelayError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        let Some(inner) = self.inner.as_mut() else {
            return self.fail();
        };
        let frame = Pin::new(inner).poll_frame(cx);
        match frame {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => {
                    if self
                        .inner
                        .as_ref()
                        .is_some_and(http_body::Body::is_end_stream)
                    {
                        self.terminalize(HttpBodyTermination::Complete);
                    }
                    Poll::Ready(Some(Ok(Frame::data(data))))
                }
                Err(_trailers) => self.fail(),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(),
            Poll::Ready(None) => {
                self.terminalize(HttpBodyTermination::Complete);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

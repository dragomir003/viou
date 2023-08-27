#![deny(missing_docs)]
#![doc = include_str!("../Readme.md")]

use std::{collections::VecDeque, sync::Arc};

use opencv::{
    core::{self as cv, Ptr},
    tracking::{TrackerKCF, TrackerKCF_Params},
    video::TrackerTrait,
};

///
#[derive(Debug, thiserror::Error)]
pub enum Error {
    ///
    #[error("Processing error occurred {0}")]
    OpenCv(#[from] opencv::Error),

    ///
    #[error("Internal logic error")]
    LogicError,
}

///
pub type Result<T> = ::core::result::Result<T, Error>;

///
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct Rect {
    xmin: isize,
    ymin: isize,
    width: isize,
    height: isize,
}

impl Rect {
    ///
    pub fn new_tlwh(xmin: isize, ymin: isize, width: isize, height: isize) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
        }
    }
}

impl From<cv::Rect> for Rect {
    fn from(value: cv::Rect) -> Self {
        Self {
            xmin: value.x as isize,
            ymin: value.y as isize,
            width: value.width as isize,
            height: value.height as isize,
        }
    }
}

impl From<Rect> for cv::Rect {
    fn from(val: Rect) -> Self {
        cv::Rect::from((
            val.xmin as i32,
            val.ymin as i32,
            val.width as i32,
            val.height as i32,
        ))
    }
}

/// Simple struct representing a detection used for updating tracks.
///
/// This would have been a trait if `impl` could be used in return type.
#[derive(Debug, Default, PartialEq, PartialOrd, Copy, Clone)]
pub struct Detection {
    bbox: Rect,
    confidence: f32,
    class: usize,
}

impl Detection {
    ///
    pub fn new(bbox: impl Into<Rect>, confidence: impl Into<f32>, class: impl Into<usize>) -> Self {
        Self {
            bbox: bbox.into(),
            confidence: confidence.into(),
            class: class.into(),
        }
    }
}

/// Simple struct representing an object that is tracked.
#[derive(Debug, Default)]
pub struct Track {
    bboxes: Vec<(Rect, bool)>,
    score: f32,
    class: usize,
    visual_tracker: Option<Ptr<TrackerKCF>>,
    ttl: usize,
    det_counter: usize,
    start_frame: usize,
}

impl Track {
    fn get_iou(&self, det: &Detection) -> f32 {
        let t: cv::Rect = self
            .bboxes
            .last()
            .map(|(r, _)| *r)
            .unwrap_or_default()
            .into();
        let d: cv::Rect = det.bbox.into();

        let i = (t | d).area() as f32;
        let u = t.area() as f32 + d.area() as f32 - i;

        i / u
    }

    fn update(mut self, det: Detection) -> Self {
        self.bboxes.push((det.bbox, true));
        if det.confidence > self.score {
            self.score = det.confidence;
            self.class = det.class;
        }
        self.det_counter += 1;
        self.ttl = 0;
        self.visual_tracker = None;

        self
    }

    fn init_visual(&mut self, frame: &cv::Mat) -> Result<()> {
        self.visual_tracker = Some(TrackerKCF::create(TrackerKCF_Params::default()?)?);
        self.visual_tracker.as_mut().unwrap().init(
            frame,
            self.bboxes
                .last()
                .map(|(r, _)| *r)
                .ok_or(Error::LogicError)?
                .into(),
        )?;
        Ok(())
    }

    fn visual_update(&mut self, frame: &cv::Mat) -> Result<bool> {
        let tracker = self.visual_tracker.as_mut().ok_or(Error::LogicError)?;

        self.ttl += 1;
        let mut roi = Default::default();
        if tracker.update(frame, &mut roi)? {
            self.bboxes.push((roi.into(), false));
            return Ok(true);
        }

        Ok(false)
    }

    ///
    pub fn new(detection: Detection, frame: usize) -> Self {
        Self {
            bboxes: vec![(detection.bbox, true)],
            score: detection.confidence,
            class: detection.class,
            visual_tracker: None,
            det_counter: 1,
            ttl: 0,
            start_frame: frame,
        }
    }
}

///
#[derive(Default)]
pub struct Tracker {
    active: Vec<Track>,
    extendable: Vec<Track>,
    finished: Vec<Track>,
    frames: VecDeque<Arc<cv::Mat>>,
    ttl: usize,
    sigma_l: f32,
    sigma_h: f32,
    sigma_iou: f32,
    t_min: usize,
    frame: usize,
}

impl Tracker {
    ///
    pub fn new(sigma_l: f32, sigma_h: f32, sigma_iou: f32, t_min: usize, ttl: usize) -> Self {
        Self {
            sigma_l,
            sigma_h,
            sigma_iou,
            t_min,
            ttl,
            ..Default::default()
        }
    }

    ///
    pub fn run(&mut self, mut detections: Vec<Detection>, frame: Arc<cv::Mat>) -> Result<()> {
        self.frame += 1;

        detections.retain(|d| d.confidence > self.sigma_l);

        if self.frames.len() > self.ttl {
            self.frames.pop_front();
        }
        self.frames.push_back(Arc::clone(&frame));

        let Association {
            matched,
            unmatched_tracks,
            unmatched_detections,
        } = associate(std::mem::take(&mut self.active), detections, self.sigma_iou);

        let mut updated = matched
            .into_iter()
            .map(|(track, detection)| track.update(detection))
            .collect::<Vec<_>>();

        for mut track in unmatched_tracks.into_iter() {
            if track.ttl > self.ttl {
                self.extendable.push(track);
                continue;
            }

            if track.ttl == 0 {
                track.init_visual(self.frames.iter().nth_back(1).ok_or(Error::LogicError)?)?;
            }

            let frame = self.frames.back().ok_or(Error::LogicError)?;

            if track.visual_update(frame)? {
                updated.push(track);
            } else {
                self.extendable.push(track);
            }
        }

        for mut track in std::mem::take(&mut self.extendable).into_iter() {
            if track.start_frame + track.bboxes.len() + track.ttl >= self.frame {
                self.extendable.push(track);
            } else if track.score >= self.sigma_h && track.det_counter >= self.t_min {
                track.visual_tracker = None;
                self.finished.push(track);
            }
        }

        self.extendable.sort_unstable_by_key(|t| t.bboxes.len());

        for detection in unmatched_detections.into_iter() {
            let mut bboxes = Vec::<Rect>::with_capacity(self.ttl);
            let mut tracker = TrackerKCF::create(TrackerKCF_Params::default()?)?;
            tracker.init(
                self.frames.back().ok_or(Error::LogicError)?.as_ref(),
                detection.bbox.into(),
            )?;

            let mut found_track = None::<usize>;

            'outer: for frame in self.frames.iter().rev().skip(1) {
                let mut bbox = Default::default();
                if !tracker.update(frame.as_ref(), &mut bbox)? {
                    break;
                }

                bboxes.push(bbox.into());

                for (i, track) in self.extendable.iter_mut().enumerate() {
                    let offset = track.start_frame + track.bboxes.len() + bboxes.len() - self.frame;

                    if 1 <= offset
                        && offset <= track.ttl
                        && track
                            .bboxes
                            .iter()
                            .nth_back(offset - 1)
                            .is_some_and(|(r, _)| {
                                let r: cv::Rect = (*r).into();
                                let d = bbox;
                                let i = (r & d).area();
                                let u = r.area() + d.area() - i;
                                (i as f32 / u as f32) >= self.sigma_iou
                            })
                    {
                        track.bboxes.drain((track.bboxes.len() - offset)..);
                        track
                            .bboxes
                            .append(&mut bboxes.iter().map(|r| (*r, false)).rev().collect());
                        let t = std::mem::take(track);
                        *track = t.update(detection);
                        found_track = Some(i);
                        break 'outer;
                    }
                }
            }

            updated.push(
                found_track
                    .map(|idx| self.extendable.remove(idx))
                    .unwrap_or(Track::new(detection, self.frame)),
            );
        }

        self.active = updated;

        Ok(())
    }
}

struct Association {
    matched: Vec<(Track, Detection)>,
    unmatched_tracks: Vec<Track>,
    unmatched_detections: Vec<Detection>,
}

fn associate(
    mut tracks: Vec<Track>,
    mut detections: Vec<Detection>,
    iou_thresh: f32,
) -> Association {
    tracks.sort_by(|t1, t2| t2.score.total_cmp(&t1.score));
    detections.sort_by(|d1, d2| d2.confidence.total_cmp(&d1.confidence));

    let mut matched = Vec::with_capacity(detections.len());
    let mut unmatched_tracks = Vec::with_capacity(tracks.len());

    for track in tracks.into_iter() {
        let (best_detection_idx, score) = detections
            .iter()
            .enumerate()
            .map(|(i, d)| (i, track.get_iou(d)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or_default();

        if score > iou_thresh {
            let det = detections[best_detection_idx];
            matched.push((track, det));
            detections[best_detection_idx].confidence = 0.0;
        } else {
            unmatched_tracks.push(track);
        }
    }

    detections.retain(|d| d.confidence > 0.0);

    Association {
        matched,
        unmatched_detections: detections,
        unmatched_tracks,
    }
}

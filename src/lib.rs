#![deny(missing_docs)]
#![doc = include_str!("../Readme.md")]

use std::collections::VecDeque;

use opencv::{
    core::{self as cv, Ptr},
    tracking::{TrackerKCF, TrackerKCF_Params, TrackerCSRT, TrackerCSRT_Params},
    video::TrackerTrait,
};

///
#[derive(Debug, thiserror::Error)]
pub enum Error {
    ///
    #[error("Processing error occurred {0}")]
    OpenCv(#[from] opencv::Error),
}

///
pub type Result<T> = ::core::result::Result<T, Error>;

///
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct Rect {
    ///
    pub xmin: isize,
    ///
    pub ymin: isize,
    ///
    pub width: isize,
    ///
    pub height: isize,
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

    ///
    pub fn get_center(&self) -> (isize, isize) {
        (self.xmin + self.width / 2, self.ymin + self.height / 2)
    }

    ///
    pub fn get_bottom_center(&self) -> (isize, isize) {
        (self.xmin + self.width / 2, self.ymin + self.height)
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
    ///
    pub bbox: Rect,
    ///
    pub confidence: f32,
    ///
    pub class: usize,
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
    ///
    pub id: u128,
    ///
    pub bboxes: Vec<Rect>,
    ///
    pub score: f32,
    ///
    pub class: usize,
    visual_tracker: Option<Ptr<TrackerCSRT>>,
    ttl: usize,
    det_counter: usize,
    ///
    pub start_frame: usize,
}

unsafe impl Send for Track {}

impl Track {
    fn get_iou(&self, det: &Detection) -> f32 {
        self.bboxes.last()
            .map_or(0.0, |b| find_iou(*b, det.bbox))
    }

    fn update(mut self, det: Detection) -> Self {
        self.bboxes.push(det.bbox);
        if det.confidence > self.score {
            self.score = det.confidence;
            self.class = det.class;
        }
        self.det_counter += 1;
        self.ttl = 0;
        if let Some(tracker) = self.visual_tracker.take() {
            drop(tracker);

        }

        self
    }

    fn init_visual(&mut self, frame: &cv::Mat) -> Result<()> {
        self.visual_tracker = Some(TrackerCSRT::create(&TrackerCSRT_Params::default()?)?);
        self.visual_tracker.as_mut().unwrap().init(
            frame,
            self.bboxes
                .last()
                .copied()
                .expect("Tried to init visual tracker on empty track")
                .into(),
        )?;
        Ok(())
    }

    fn visual_update(&mut self, frame: &cv::Mat) -> Result<bool> {
        let tracker = self.visual_tracker.as_mut().expect("Failed to init visual tracker");

        self.ttl += 1;
        let mut roi = Default::default();
        if tracker.update(frame, &mut roi)? {
            self.bboxes.push(roi.into());
            return Ok(true);
        }

        Ok(false)
    }

    ///
    pub fn new(id: impl Into<u128>, detection: Detection, frame: usize) -> Self {
        Self {
            bboxes: vec![detection.bbox],
            score: detection.confidence,
            class: detection.class,
            visual_tracker: None,
            det_counter: 1,
            ttl: 0,
            start_frame: frame,
            id: id.into(),
        }
    }

    fn finish(mut self) -> Track {
        self.visual_tracker = None;
        self
    }
}

///
#[derive(Default)]
pub struct Tracker {
    active: Vec<Track>,
    extendable: Vec<Track>,
    finished: Vec<Track>,
    frames: VecDeque<cv::Mat>,
    ttl: usize,
    sigma_l: f32,
    sigma_h: f32,
    sigma_iou: f32,
    t_min: usize,
    frame: usize,
    last_id: u128,
}


unsafe impl Send for Tracker {}

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
    pub fn finish(mut self) -> Vec<Track> {
        self.active.append(&mut self.finished);
        self.extendable.retain(|t| t.score >= self.sigma_h && t.det_counter >= self.t_min);
        self.active.append(&mut self.extendable);

        self.active
    }

    ///
    pub fn run(&mut self, mut detections: Vec<Detection>, frame: cv::Mat) -> Result<(&[Track], &[Track])> {
        self.frame += 1;

        detections.retain(|d| d.confidence > self.sigma_l);

        if self.frames.len() > self.ttl {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);

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
                track.init_visual(self.frames.iter().nth_back(1).expect("Tried to visualy track with no frames"))?;
            }

            let frame = self.frames.back().expect("There must be frames in tracker");

            if track.visual_update(frame)? {
                updated.push(track);
            } else {
                self.extendable.push(track);
            }
        }

        for track in std::mem::take(&mut self.extendable).into_iter() {
            if track.start_frame + track.bboxes.len() - track.ttl + self.ttl >= self.frame {
                self.extendable.push(track);
            } else if track.score >= self.sigma_h && track.det_counter >= self.t_min {
                self.finished.push(track.finish());
            }
        }

        self.extendable.sort_unstable_by_key(|t| t.bboxes.len());

        for detection in unmatched_detections.into_iter() {
            let mut bboxes = Vec::<Rect>::with_capacity(self.ttl);
            let mut tracker = TrackerCSRT::create(&TrackerCSRT_Params::default()?)?;
            tracker.init(
                self.frames.back().expect("Ther must have been a frame by now"),
                detection.bbox.into(),
            )?;

            let mut found_track = None::<usize>;

            'outer: for (frame_idx, frame) in self.frames.iter().rev().enumerate().skip(1) {
                let mut bbox = Default::default();
                if !tracker.update(frame, &mut bbox)? {
                    break;
                }

                bboxes.push(bbox.into());

                for (i, track) in self.extendable.iter_mut().enumerate() {
                    let tbox = if track.start_frame + track.bboxes.len() < self.frame - frame_idx {
                        continue;
                    } else {
                        let Some(bbox) = track.bboxes.iter().nth_back(frame_idx - 1) else {
                            continue;
                        };
                        bbox
                    };

                    if find_iou(*tbox, bbox) >= self.sigma_iou {
                        track.bboxes.drain((track.bboxes.len() - frame_idx)..);
                        track.bboxes.append(&mut bboxes);
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
                    .unwrap_or_else(|| {
                        self.last_id += 1;
                        Track::new(self.last_id - 1, detection, self.frame)
                    })
            );
        }

        self.active = updated;


        Ok((self.active.as_slice(), self.get_extendable_tracks()))
    }

    ///
    pub fn get_extendable_tracks(&self) -> &[Track] {
        &self.extendable
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

fn find_iou(a: impl Into<cv::Rect>, b: impl Into<cv::Rect>) -> f32 {
    let a = a.into();
    let b = b.into();

    let i = (a & b).area() as f32;
    let u = (a.area() + b.area()) as f32 - i;

    i / u
}

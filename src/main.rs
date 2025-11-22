use image::{GrayImage, ImageReader, Luma};
use imageproc::distance_transform::Norm;
use imageproc::edges::canny;
use imageproc::morphology::dilate;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use macroquad::prelude::*;

const WINDOW_WIDTH: i32 = 900;
const WINDOW_HEIGHT: i32 = 800;
const NUM_EPICYCLE_TERMS: usize = 500;
const ANIMATION_FRAMES: f32 = 1200.0;
const MAX_PATH_POINTS: usize = 15000;
const CANNY_LOW_THRESHOLD: f32 = 50.0;
const CANNY_HIGH_THRESHOLD: f32 = 100.0;
const DILATE_RADIUS: u8 = 2;
const MIN_EPICYCLE_AMPLITUDE: f32 = 0.001;
const EDGE_POINT_RADIUS: f32 = 1.0;
const EPICYCLE_LINE_WIDTH: f32 = 1.0;
const PATH_LINE_WIDTH: f32 = 2.0;

type Point2D = (f32, f32);

fn window_conf() -> Conf {
    Conf {
        window_title: "Fourier Epicycle Drawer".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        ..Default::default()
    }
}


#[macroquad::main(window_conf)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "shape_0.png";
    let edge_points = get_edge_points(path)?;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(edge_points.len());

    let mut buffer: Vec<Complex<f32>> = edge_points
        .iter()
        .map(|&(x, y)| Complex::new(x, y))
        .collect();

    fft.process(&mut buffer);

    let epicycles = fft_to_epicycles(&buffer, NUM_EPICYCLE_TERMS);

    let mut time = 0.0;
    let mut path_points: Vec<Vec2> = Vec::new();
    let dt = 2.0 * std::f32::consts::PI / ANIMATION_FRAMES;

    loop {
        draw_frame(&edge_points, &epicycles, &mut time, &mut path_points, dt);
        next_frame().await
    }
}

fn draw_frame(
    edge_points: &[Point2D],
    epicycles: &[Epicycle],
    time: &mut f32,
    path_points: &mut Vec<Vec2>,
    dt: f32,
) {
    clear_background(BLACK);

    let center = vec2(screen_width() / 2.0, screen_height() / 2.0);

    // Draw original edge points as reference (faint gray dots)
    for &(x, y) in edge_points {
        draw_circle(center.x + x, center.y + y, EDGE_POINT_RADIUS, DARKGRAY);
    }

    // Draw info text
    draw_text(&format!("Time: {:.2}", *time), 20.0, 20.0, 30.0, WHITE);
    draw_text(
        &format!("Epicycles: {}", epicycles.len()),
        20.0,
        50.0,
        30.0,
        WHITE,
    );
    draw_text(
        &format!("Edge Points: {}", edge_points.len()),
        20.0,
        80.0,
        30.0,
        WHITE,
    );

    // Calculate current point by summing all epicycles
    let mut current_pos = center;

    for epicycle in epicycles {
        let prev_pos = current_pos;
        let radius = epicycle.amp;
        let angle = epicycle.freq * *time + epicycle.phase;
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        current_pos += vec2(x, y);

        // Draw epicycle circle (semi-transparent)
        draw_circle_lines(
            prev_pos.x,
            prev_pos.y,
            radius,
            EPICYCLE_LINE_WIDTH,
            Color::new(0.5, 0.5, 0.5, 0.3),
        );

        // Draw line from circle center to current position
        draw_line(
            prev_pos.x,
            prev_pos.y,
            current_pos.x,
            current_pos.y,
            EPICYCLE_LINE_WIDTH,
            WHITE,
        );
    }

    // Add current point to path (limit size to prevent memory issues)
    if path_points.len() > MAX_PATH_POINTS {
        path_points.remove(0);
    }
    path_points.push(current_pos);

    // Draw the traced path
    for i in 0..path_points.len().saturating_sub(1) {
        draw_line(
            path_points[i].x,
            path_points[i].y,
            path_points[i + 1].x,
            path_points[i + 1].y,
            PATH_LINE_WIDTH,
            YELLOW,
        );
    }

    // Reset animation after full cycle
    if *time > 2.0 * std::f32::consts::PI {
        *time = 0.0;
        path_points.clear();
    }

    *time += dt;
}

fn get_edge_points(path: &str) -> Result<Vec<Point2D>, Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)
        .map_err(|e| format!("Failed to open image '{}': {}", path, e))?
        .decode()
        .map_err(|e| format!("Failed to decode image '{}': {}", path, e))?
        .to_luma8();

    let (width, height) = img.dimensions();
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;

    // 1. Canny Edge Detection
    let edges = canny(&img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);

    // 2. Dilate to merge double edges
    let dilated = dilate(&edges, Norm::LInf, DILATE_RADIUS);

    // 3. Skeletonize to get single centerline
    let skeleton = skeletonize(&dilated);

    // 4. Extract non-zero pixels as points (centered at origin)
    let mut points = Vec::new();
    for (x, y, pixel) in skeleton.enumerate_pixels() {
        if pixel[0] > 0 {
            points.push((x as f32 - center_x, y as f32 - center_y));
        }
    }

    Ok(sort_points(points))
}

/// Represents the 8 neighboring pixels around a center pixel
struct Neighbors {
    p2: u8, // North
    p3: u8, // North-East
    p4: u8, // East
    p5: u8, // South-East
    p6: u8, // South
    p7: u8, // South-West
    p8: u8, // West
    p9: u8, // North-West
}

impl Neighbors {
    /// Gets the 8 neighbors of a pixel at (x, y) in the image
    fn get(img: &GrayImage, x: u32, y: u32) -> Self {
        Self {
            p2: u8::from(img.get_pixel(x, y - 1)[0] > 0),
            p3: u8::from(img.get_pixel(x + 1, y - 1)[0] > 0),
            p4: u8::from(img.get_pixel(x + 1, y)[0] > 0),
            p5: u8::from(img.get_pixel(x + 1, y + 1)[0] > 0),
            p6: u8::from(img.get_pixel(x, y + 1)[0] > 0),
            p7: u8::from(img.get_pixel(x - 1, y + 1)[0] > 0),
            p8: u8::from(img.get_pixel(x - 1, y)[0] > 0),
            p9: u8::from(img.get_pixel(x - 1, y - 1)[0] > 0),
        }
    }

    /// Counts the number of 0->1 transitions in the ordered sequence of neighbors
    fn count_transitions(&self) -> u8 {
        let transitions = [
            (self.p2, self.p3),
            (self.p3, self.p4),
            (self.p4, self.p5),
            (self.p5, self.p6),
            (self.p6, self.p7),
            (self.p7, self.p8),
            (self.p8, self.p9),
            (self.p9, self.p2),
        ];

        transitions
            .iter()
            .filter(|&&(prev, curr)| prev == 0 && curr == 1)
            .count() as u8
    }

    /// Counts the total number of non-zero neighbors
    fn count_nonzero(&self) -> u8 {
        self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9
    }
}

/// Applies Zhang-Suen thinning algorithm to skeletonize the image
fn skeletonize(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut current = img.clone();
    let mut changed = true;

    while changed {
        changed = false;
        let mut next = current.clone();
        let mut to_clear = Vec::new();

        // Step 1: Check first set of conditions
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if current.get_pixel(x, y)[0] == 0 {
                    continue;
                }

                let n = Neighbors::get(&current, x, y);
                let a = n.count_transitions();
                let b = n.count_nonzero();

                if a == 1 && b >= 2 && b <= 6 && n.p2 * n.p4 * n.p6 == 0 && n.p4 * n.p6 * n.p8 == 0
                {
                    to_clear.push((x, y));
                    changed = true;
                }
            }
        }

        for &(x, y) in &to_clear {
            next.put_pixel(x, y, Luma([0]));
        }
        current = next.clone();
        to_clear.clear();

        // Step 2: Check second set of conditions
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if current.get_pixel(x, y)[0] == 0 {
                    continue;
                }

                let n = Neighbors::get(&current, x, y);
                let a = n.count_transitions();
                let b = n.count_nonzero();

                if a == 1 && b >= 2 && b <= 6 && n.p2 * n.p4 * n.p8 == 0 && n.p2 * n.p6 * n.p8 == 0
                {
                    to_clear.push((x, y));
                    changed = true;
                }
            }
        }

        for (x, y) in to_clear {
            next.put_pixel(x, y, Luma([0]));
        }
        current = next;
    }

    current
}

fn sort_points(mut points: Vec<Point2D>) -> Vec<Point2D> {
    if points.is_empty() {
        return points;
    }

    let mut sorted_points = Vec::with_capacity(points.len());
    let mut current_point = points.remove(0);
    sorted_points.push(current_point);

    while !points.is_empty() {
        let mut nearest_idx = 0;
        let mut min_dist_sq = f32::MAX;

        for (i, &point) in points.iter().enumerate() {
            let dist_sq = (current_point.0 - point.0).powi(2) + (current_point.1 - point.1).powi(2);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        current_point = points.swap_remove(nearest_idx);
        sorted_points.push(current_point);
    }

    sorted_points
}

/// Represents a rotating circle (epicycle) in the Fourier series
#[derive(Debug)]
struct Epicycle {
    freq: f32,
    amp: f32,
    phase: f32,
}

/// Converts FFT output to a sorted list of epicycles
fn fft_to_epicycles(buffer: &[Complex<f32>], num_terms: usize) -> Vec<Epicycle> {
    let n = buffer.len();
    let mut epicycles: Vec<Epicycle> = buffer
        .iter()
        .enumerate()
        .map(|(k, val)| {
            // Convert frequency index to signed frequency
            let freq = if k <= n / 2 {
                k as f32
            } else {
                (k as i32 - n as i32) as f32
            };
            let amp = val.norm() / n as f32;
            let phase = val.arg();
            Epicycle { freq, amp, phase }
        })
        .filter(|e| e.amp > MIN_EPICYCLE_AMPLITUDE)
        .collect();

    // Sort by amplitude (largest first) and keep only top terms
    epicycles.sort_by(|a, b| b.amp.partial_cmp(&a.amp).unwrap());
    epicycles.truncate(num_terms);
    epicycles
}

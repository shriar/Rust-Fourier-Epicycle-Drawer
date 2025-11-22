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
    let img = ImageReader::open(path)?.decode()?.to_luma8();
    let (width, height) = img.dimensions();
    let center = (width as f32 / 2.0, height as f32 / 2.0);

    let edges = canny(&img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
    let dilated = dilate(&edges, Norm::LInf, DILATE_RADIUS);
    let skeleton = skeletonize(&dilated);

    let points: Vec<Point2D> = skeleton
        .enumerate_pixels()
        .filter(|(_, _, p)| p[0] > 0)
        .map(|(x, y, _)| (x as f32 - center.0, y as f32 - center.1))
        .collect();

    Ok(sort_points(points))
}

fn get_neighbors(img: &GrayImage, x: u32, y: u32) -> [u8; 8] {
    [
        u8::from(img.get_pixel(x, y - 1)[0] > 0),     // N
        u8::from(img.get_pixel(x + 1, y - 1)[0] > 0), // NE
        u8::from(img.get_pixel(x + 1, y)[0] > 0),     // E
        u8::from(img.get_pixel(x + 1, y + 1)[0] > 0), // SE
        u8::from(img.get_pixel(x, y + 1)[0] > 0),     // S
        u8::from(img.get_pixel(x - 1, y + 1)[0] > 0), // SW
        u8::from(img.get_pixel(x - 1, y)[0] > 0),     // W
        u8::from(img.get_pixel(x - 1, y - 1)[0] > 0), // NW
    ]
}

fn count_transitions(n: &[u8; 8]) -> u8 {
    (0..8).filter(|&i| n[i] == 0 && n[(i + 1) % 8] == 1).count() as u8
}

fn should_remove(n: &[u8; 8], step: u8) -> bool {
    let transitions = count_transitions(n);
    let nonzero: u8 = n.iter().sum();
    let condition = transitions == 1 && (2..=6).contains(&nonzero);

    if step == 1 {
        condition && n[0] * n[2] * n[4] == 0 && n[2] * n[4] * n[6] == 0
    } else {
        condition && n[0] * n[2] * n[6] == 0 && n[0] * n[4] * n[6] == 0
    }
}

fn skeletonize(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut current = img.clone();
    let mut changed = true;

    while changed {
        changed = false;

        for step in 1..=2 {
            let mut to_clear = Vec::new();

            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    if current.get_pixel(x, y)[0] > 0 {
                        let neighbors = get_neighbors(&current, x, y);
                        if should_remove(&neighbors, step) {
                            to_clear.push((x, y));
                            changed = true;
                        }
                    }
                }
            }

            for (x, y) in to_clear {
                current.put_pixel(x, y, Luma([0]));
            }
        }
    }

    current
}

fn dist_sq(p1: Point2D, p2: Point2D) -> f32 {
    (p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)
}

fn sort_points(mut points: Vec<Point2D>) -> Vec<Point2D> {
    if points.is_empty() {
        return points;
    }

    let mut sorted = Vec::with_capacity(points.len());
    let mut current = points.remove(0);
    sorted.push(current);

    while !points.is_empty() {
        let (idx, _) = points
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                dist_sq(current, a)
                    .partial_cmp(&dist_sq(current, b))
                    .unwrap()
            })
            .unwrap();

        current = points.swap_remove(idx);
        sorted.push(current);
    }

    sorted
}

#[derive(Debug)]
struct Epicycle {
    freq: f32,
    amp: f32,
    phase: f32,
}

fn fft_to_epicycles(buffer: &[Complex<f32>], num_terms: usize) -> Vec<Epicycle> {
    let n = buffer.len();
    let mut epicycles: Vec<Epicycle> = buffer
        .iter()
        .enumerate()
        .filter_map(|(k, val)| {
            let amp = val.norm() / n as f32;
            if amp > MIN_EPICYCLE_AMPLITUDE {
                let freq = if k <= n / 2 {
                    k as f32
                } else {
                    (k as i32 - n as i32) as f32
                };
                Some(Epicycle {
                    freq,
                    amp,
                    phase: val.arg(),
                })
            } else {
                None
            }
        })
        .collect();

    epicycles.sort_by(|a, b| b.amp.partial_cmp(&a.amp).unwrap());
    epicycles.truncate(num_terms);
    epicycles
}

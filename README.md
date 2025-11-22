# Fourier Epicycle Drawer

A Rust application that uses Fourier transforms to animate drawings using epicycles.

## Description

This project takes an input image, extracts its edges, and uses Fast Fourier Transform (FFT) to decompose the shape into rotating circles (epicycles). The epicycles are then animated to trace out the original shape, creating a mesmerizing visualization.

## Dependencies

- `image` - Image loading and processing
- `imageproc` - Edge detection and morphological operations
- `rustfft` - Fast Fourier Transform implementation
- `macroquad` - Graphics and windowing

## Usage

1. Place your input image (PNG format) in the project directory
2. Update the image path in `main.rs` (line 37):
   ```rust
   let path = "shape_0.png"; // Replace with your image path
   ```
3. Run the project:
   ```bash
   cargo run --release
   ```

## How It Works

1. **Edge Extraction**: Loads an image and applies Canny edge detection
2. **Preprocessing**: Dilates edges and applies skeletonization (Zhang-Suen algorithm) to get clean contours
3. **Point Sorting**: Orders edge points using nearest-neighbor heuristic for continuous paths
4. **FFT Transform**: Converts spatial points to frequency domain
5. **Epicycle Generation**: Extracts top frequency components as rotating circles
6. **Animation**: Renders epicycles rotating and tracing the original shape

## Configuration

You can adjust constants at the top of `main.rs`:

- **Number of Epicycles**: `NUM_EPICYCLE_TERMS` (default: 500) - Higher values = more detail
- **Animation Frames**: `ANIMATION_FRAMES` (default: 1200) - Higher values = slower animation
- **Window Size**: `WINDOW_WIDTH` and `WINDOW_HEIGHT` (default: 900x800)
- **Edge Detection**: `CANNY_LOW_THRESHOLD` and `CANNY_HIGH_THRESHOLD` for sensitivity

## License

MIT

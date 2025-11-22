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
2. Update the image path in `main.rs` (line 22):
   ```rust
   let path = "shape_0.png"; // Replace with your image path
   ```
3. Run the project:
   ```bash
   cargo run
   ```

## How It Works

1. **Edge Extraction**: Loads an image and applies Canny edge detection
2. **Preprocessing**: Dilates edges and applies skeletonization to get clean contours
3. **Point Sorting**: Orders edge points using nearest-neighbor heuristic
4. **FFT Transform**: Converts spatial points to frequency domain
5. **Epicycle Generation**: Extracts top frequency components as rotating circles
6. **Animation**: Renders epicycles rotating and tracing the original shape

## Configuration

- **Number of Epicycles**: Adjust `num_terms` parameter (line 35) for more/less detail
- **Animation Speed**: Modify `dt` calculation (line 41) for faster/slower animation
- **Window Size**: Change dimensions in `window_conf()` function (lines 14-15)

## License

MIT

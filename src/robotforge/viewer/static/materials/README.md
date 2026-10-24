# Bundled ground image

Oak Wood Planks by Dimitrios Savva, from [Poly Haven](https://polyhaven.com/a/oak_wood_planks).
The original asset is [CC0-1.0](https://polyhaven.com/license); this image data may be redistributed and modified.
Its nominal width is 1.2 meters, matching the default wood ground's existing tile scale.
The app makes no request to Poly Haven and needs no API account.

Only derived 512-square grayscale albedo and 256-square packed height/roughness images are embedded in `../ground-wood.js`.
Grayscale preserves the world's authored colour tint; its decoded mean linear factor stays within 1 percent of the prior procedural wood.
Height uses red and roughness uses green, with no additional procedural joint grid.
The module loads only for used wood ground or an explicit wood thumbnail, with a bounded procedural fallback.
Image data totals 86,881 bytes; the two maps retain their existing shader slots and mipmapped filtering.
The 512 colour / 256 detail combination adds about 1 MiB nominal GPU pixel storage relative to two 256 maps; CPU/driver allocations are separate.

For offline reproduction, download these official 1k JPGs into a directory:

- [Diffuse](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/oak_wood_planks/oak_wood_planks_diff_1k.jpg)
- [Displacement](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/oak_wood_planks/oak_wood_planks_disp_1k.jpg)
- [Roughness](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/oak_wood_planks/oak_wood_planks_rough_1k.jpg)

Run `uv run --frozen python scripts/prepare_ground_wood.py SOURCE_DIR --output src/robotforge/viewer/static/ground-wood.js`.
The script verifies their exact SHA256 checksums, linear-light BOX filtering, fixed processing parameters, decoded mean and 128 KiB image/module caps.
Use the repository's locked Pillow version for byte reproduction.
Full originals and rendered previews are outside the package; the license applies to the image data, while the loader follows the project's code license.

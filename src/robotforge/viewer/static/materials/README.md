# Bundled ground images

## Oak

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

## Grass

Leafy Grass by Charlotte Baglioni, from [Poly Haven](https://polyhaven.com/a/leafy_grass), is [CC0-1.0](https://polyhaven.com/license).
The installed app uses two derived 256-square images in `../ground-grass.js`, with no runtime asset service or new dependency.
Grayscale albedo preserves the authored ground tint; packed detail reads height from red and roughness from green.
Image data totals 77,254 bytes, with the same two 256-square GPU map slots, shader and filtering as procedural grass.
Nominal GPU pixel storage is unchanged; CPU and driver costs must be measured separately.
The source's nominal 2 meter width is artistically reduced to the existing 0.6 meter tile, retaining every scene's UV mapping rather than claiming physical leaf scale.

For offline reproduction, download these official 1k JPGs into a directory:

| Channel | Source | SHA256 |
| --- | --- | --- |
| Diffuse | [JPG](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/leafy_grass/leafy_grass_diff_1k.jpg) | `cfa40bc9d9417d1852db8753a8d5917f110c40101179f63543c382e39bc05e4a` |
| Displacement | [JPG](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/leafy_grass/leafy_grass_disp_1k.jpg) | `3c78000885796046cc90cec2a61abea0f65cd8fb6585ac2a7b5987f73c5aa98e` |
| Roughness | [JPG](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/leafy_grass/leafy_grass_rough_1k.jpg) | `0c2a1d91aacb8e54a496051b59854e51e456fe858abeade778e308359ef16463` |

Run `uv run --frozen python scripts/prepare_ground_grass.py SOURCE_DIR --output src/robotforge/viewer/static/ground-grass.js`.
The fixed recipe uses linear-light BOX filtering, half relative albedo contrast, target linear mean 0.8164235380873729, height mean 0.4333371629901961 with gain 0.4 and roughness mean 0.9797357297411152 with gain 0.1.
The script checks source hashes and dimensions, decoded mean within 1 percent, a 96 KiB combined image cap and a 128 KiB module cap.
The derived albedo SHA256 is `5c76596cd17b2b516514a8ec4068e88ea8b0d07d4988af88c47b6574a5fde1ed` and detail SHA256 is `b0db64526e3b0f76c7308ebf1d8e92f6943ea154c3977ccc053a7dc5ced108f8`.
Only used grass geometry or an explicit detailed thumbnail loads the module.
Wood and grass prepare concurrently with independent atomic pairs and 1000 ms optional deadlines; failures use procedural fallback, while late images only fill the CPU cache for later scenes.
Full originals and verification videos stay outside the package.

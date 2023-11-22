# DX7-JAX

This is a work-in-progress of using [JAX](https://jax.readthedocs.io/en/latest/) to batch-render [Yamaha DX7](https://en.wikipedia.org/wiki/Yamaha_DX7) presets. We also provide a script to parse `.syx` files and aggregate them into a single CSV.

The [Faust](https://faust.grame.fr/) Libraries have an [implementation](https://faustlibraries.grame.fr/libs/dx7/#dxdx7_ui) of the DX7, but it lacks several features. We have improved it in several ways such as adding an LFO, **but our implementation (`custom_dx7.lib`) is imperfect, so please use this project with caution.** Pull requests are welcome.

DX7-JAX uses [DawDreamer](https://github.com/DBraun/DawDreamer/) to convert the Faust code to JAX, which can then be executed on either the CPU or GPU. We hope this project can be a useful starter for research involving Faust and JAX, not necessarily involving the DX7. DawDreamer has other JAX [examples](https://github.com/DBraun/DawDreamer/blob/main/examples/Faust_to_JAX/Faust_to_JAX.ipynb), which we encourage you to check out.

## Download patches

We mirror the `DX7_AllTheWeb.zip` (2023-08-08) from <http://bobbyblues.recup.ch/yamaha_dx7/dx7_patches.html> (saving them some bandwidth). Use this script to download it to the right location.

```bash
python download_patches.py
```

## Turn patches into a CSV

This will parse a limited amount of presets.

```bash
python parse_dx7.py --directory dx7_patches/DX7_AllTheWeb/Atari
```

This will parse all presets. You probably shouldn't do this because the later rendering output will be enormous.

```bash
python parse_dx7.py
```

If you run it on all of `DX7_AllTheWeb`, then 388,650 presets will be de-duplicated into 44,884.

## Render an audio dataset

Render the audio with default args (`python dx7_render.py --help` for help)

```bash
python dx7_render.py
```

## Use our DX7 in the Faust IDE.

Go to the [Faust IDE](https://faustide.grame.fr) and paste the content of `custom_dx7.lib` into the text editor. Then at the bottom, paste

`process = dx7_algorithm(1) <: _, _;`

This selects the *1st* of the 32 DX7 algorithms. Enable "Poly Voices" on the left hand side. Four is a good number. Then press the play button towards the top of the screen and play around with your keyboard as the controller. 

## Citation

```
@software{Braun_DX7-JAX_2023,
    author = {Braun, David},
    month = nov,
    title = {{DX7-JAX}},
    url = {https://github.com/DBraun/DX7-JAX},
    version = {0.0.1},
    year = {2023}
}
```

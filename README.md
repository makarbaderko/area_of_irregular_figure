# Problem

Find area of an irregular shape, by photo.

## Installation
Clone the repository

```
git clone https://github.com/makarbaderko/area_of_irregular_figure
cd area_of_irregular_figure
```

Install all needed libraries
```
pip install -r requirements.txt
```

## Usage

Make a photo, of an irregular figure, where the figure is in the center and the ruler starts in the left side of the screen and ends in the right side of the screen. Then, run the code!
```
python3 main.py
```

## How does it work?

We delete the background by the big colour diference between the figure and the background and save the image in RGBA format.
After that, we count all non-alpha pixels (their A value is >= 0), get the ruler's length.
Because we know the width of the screen, we can divide the ruler's length by the number of pixels in the width. 
Then, we get the area of one pixel in cm2, by squaring the side of a pixel.
Here, we multiply the number of non-alpha pixels, by the area of one pixel.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Nobody can copy or use my code without my direct written and signed permission.
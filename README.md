# surfaceRegistration
Tools for surface registratoin of spherical surfaces with the general elastic metric based on the work by Zhe Su, Martin Bauer, Eric Klassen and Kyle Gallivan.

## What is it?

This code provides tools for geometric shape analysis on spherical surfaces with the general elastic metric. It is able to factor out reparametrizations and translations.

For details we refer to our paper

```css
@article{su2020simplifying,
	author = {Zhe Su and Martin Bauer and Eric Klassen and Kyle Gallivan},
	title = {Simplifying Transformations for a Family of Elastic Metrics on the Space of Surfaces},
	journal = {2020 IEEE Conference on Computer Vision and Pattern Recognition Workshops},
	year = 2020,
}
```

If you use our code in your work please cite our paper.

## Packages

Please install the following packages

* Pytorch: [https://pytorch.org/](https://pytorch.org/)
* Numpy: [https://numpy.org/](https://numpy.org/)
* Scipy: [https://www.scipy.org/](https://www.scipy.org/)
* Mayavi (for plotting): [https://docs.enthought.com/mayavi/mayavi/](https://docs.enthought.com/mayavi/mayavi/)

The code was tested on jupyter notebook.

## Usage

See the file "Surface_Registration" for examples of how to use the code. Each surface should be represented using spherical coordinates as a function f:[0, &pi;] x [0, 2&pi;] &rarr; R<sup>3</sup> 
such that f(0, &theta;) = f(0, 0), f(&pi;, &theta;) = f(&pi;, 0) and f(&phi;, 0) = f(&phi;, 2&pi;). The code can be used directly for surfaces with resolution 50 x 99 (the numbers of discrete polar and azimuthal angles) since the corresponding bases are preloaded in the folder "Bases".  For the other resolutions, one can resample the surfaces or generate the corresponding bases for surfaces using the file in the folder "generateBases" and then put the bases files in the folder "Bases". 

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)

## Contacts

* Martin Bauer (bauer at math dot fsu dot edu)
* Zhe Su (zsu at math dot fsu dot edu)

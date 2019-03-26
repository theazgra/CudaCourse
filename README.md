## Homework
- Finish the terrain to normal map conversion
- [ ] Convolution (sobel operator)
- [ ] Normalization of values

## Project - Cellular Genetic Algorithm
- Fitness value will be stored in 2D Texture (`N` x `M`)
- Cells will be stored in pitched memory
  - At start `X` random cells will be generated in `N` x `M` grid
- At each evolution step
  - Every cell will look in its neighborhood for best cells (best fitness value)
    - Neighbors will be sorted according to their fitness values
  - 2 best cells will be selected and offspring will be created by some mutation method
  - Replacement method will be choosen to replace some cell with new offspring
  - Old population is replaced with new population and population fitness is calculated with parallel reduction.

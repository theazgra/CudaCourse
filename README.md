## Homework
- Finish the terrain to normal map conversion
- [x] Convolution (sobel operator)
- [x] Normalization of values

## Project - Cellular Genetic Algorithm
- Fitness value will be stored in 2D Texture (`N` x `M`). Generated gradient image.
- Cells will live in toroidal grid which will probably be stored in 2D pitched memory. Texture would be better because of 
how tex2D lookup is wrapped around, but this texture has to be reset every evolution step.
  - Each cell will have its random generated X and Y position, representing its genes. This X and Y position is not the position of the cell in toroidal grid.
  - At start full toroidal grid of cells will be generated. Random generating its genes (X and Y positions)
- At each evolution step every cell will:
  - Collect the cells in its neighborhood.(neighborhood could be modified compile time not runtime?)
  - Order neighborhood based on theirs fitness value
  - Select two best parents
  - Create offspring be recombination of parents genes.
  - Replace one parent with the offspring, save it in new population
- New population replaces the old one
- Population fitness value is calculated by reduction.

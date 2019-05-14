#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>
#include <cstring>
#include <string>
#include <limits>

constexpr int ThreadsPerBlock = 32;
constexpr int NumberOfEvolutions = 5000;
constexpr int CellGridDimension = 2048 * 4; //10000
constexpr size_t CellCount = CellGridDimension * CellGridDimension;

#define ENABLE_OGL 0

#if ENABLE_OGL

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

typedef unsigned char uchar;
constexpr int BlockDim = 32;
constexpr ushort MaxFitness = std::numeric_limits<ushort>::max();
int viewportWidth = CellGridDimension;
int viewportHeight = CellGridDimension;

uchar *renderTextureData;

unsigned int pboId;
unsigned int textureId;
cudaGraphicsResource_t cudaPBOResource;
cudaGraphicsResource_t cudaTexResource;

CellGrid grid;

float fitness = 0.0;
float lastFitness = -1.0f;
uint iter = 0;
double averageEvolveTime = 0.0f;
double averageFitnessTime = 0.0f;
size_t sameFitnessValue = 0;

void my_display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureId);

    glBegin(GL_QUADS);

    glTexCoord2d(0, 0);
    glVertex2d(0, 0);
    glTexCoord2d(1, 0);
    glVertex2d(viewportWidth, 0);
    glTexCoord2d(1, 1);
    glVertex2d(viewportWidth, viewportHeight);
    glTexCoord2d(0, 1);
    glVertex2d(0, viewportHeight);

    glEnd();

    glDisable(GL_TEXTURE_2D);

    glFlush();
    glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
    viewportWidth = w;
    viewportHeight = h;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, viewportWidth, viewportHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, viewportWidth, 0, viewportHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

__global__ void clear_kernel(unsigned char *pbo)
{
    uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint strideX = blockDim.x * gridDim.x;
    uint strideY = blockDim.y * gridDim.y;

    while (tIdX < CellGridDimension)
    {
        tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
        while (tIdY < CellGridDimension)
        {
            pbo[(tIdY * CellGridDimension * 4) + tIdX] = 0;
            pbo[(tIdY * CellGridDimension * 4) + tIdX + 1] = 0;
            pbo[(tIdY * CellGridDimension * 4) + tIdX + 2] = 0;
            pbo[(tIdY * CellGridDimension * 4) + tIdX + 3] = 0;
            tIdY += strideY;
        }

        tIdX += strideX;
    }
}

__global__ void draw(const unsigned int pboWidth, const unsigned int pboHeight, unsigned char *pbo, Cell *population)
{
    for (int cellX = 0; cellX < CellGridDimension; cellX++)
    {
        for (int cellY = 0; cellY < CellGridDimension; cellY++)
        {
            Cell cell = population[(cellY * CellGridDimension) + cellX];

            float p = cell.fitness / static_cast<float>(MaxFitness);

            pbo[(cell.x * CellGridDimension * 4) + cell.y] = 255;
            pbo[(cell.x * CellGridDimension * 4) + cell.y + 1] = 255;
            pbo[(cell.x * CellGridDimension * 4) + cell.y + 2] = 255;
            pbo[(cell.x * CellGridDimension * 4) + cell.y + 3] = 0; //(uchar)(255 * p);

            // pbo[(cell.y * CellGridDimension * 4) + cell.x] = static_cast<uchar>(255 * p);
            // pbo[(cell.y * CellGridDimension * 4) + cell.x + 1] = static_cast<uchar>(125 * p);
            // pbo[(cell.y * CellGridDimension * 4) + cell.x + 2] = static_cast<uchar>(255 * p);
            // pbo[(cell.y * CellGridDimension * 4) + cell.x + 3] = 255;
        }
    }
}

void cudaWorker()
{
    if (iter < NumberOfEvolutions && sameFitnessValue < 5)
    {
        float evolveTime, fitnessTime;

        lastFitness = fitness;
        ++iter;

        grid.evolve(evolveTime);
        averageEvolveTime += evolveTime;

        fitness = grid.get_average_fitness(fitnessTime);
        averageFitnessTime += fitnessTime;

        if (fitness == lastFitness)
            sameFitnessValue++;
        else
            sameFitnessValue = 0;

        printf("Finished iteration %u, fitness: %.6f\n", iter + 1, fitness);

        unsigned char *pboData;
        size_t pboSize;
        CUDA_CALL(cudaGraphicsMapResources(1, &cudaPBOResource, 0));

        CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)&pboData, &pboSize, cudaPBOResource));

        KernelSettings ks = grid.get_kernel_settings();
        Cell *cellMemory = grid.get_device_population_memory();

        clear_kernel<<<ks.gridDimension, ks.blockDimension>>>(pboData);

        draw<<<1, 1>>>(CellGridDimension, CellGridDimension, pboData, cellMemory);

        CUDA_CALL(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, CellGridDimension, CellGridDimension, GL_BGRA, GL_UNSIGNED_BYTE, NULL); //Source parameter is NULL, Data is coming from a PBO, not host memory
    }
}

void my_idle()
{
    cudaWorker();
    glutPostRedisplay();
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(viewportWidth, viewportHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Cellular genetic algorithm");

    glutDisplayFunc(my_display);
    glutReshapeFunc(my_resize);
    glutIdleFunc(my_idle);
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);

    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glShadeModel(GL_SMOOTH);
    glViewport(0, 0, viewportWidth, viewportHeight);

    glFlush();
}

void create_render_texture()
{
    size_t allocSize = sizeof(uchar) * 4 * CellGridDimension * CellGridDimension;
    renderTextureData = static_cast<uchar *>(::operator new(allocSize));
    memset(renderTextureData, 0, allocSize);

    //OpenGL Texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);

    //WARNING: Just some of inner format are supported by CUDA!!!
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CellGridDimension, CellGridDimension, 0, GL_RGBA, GL_UNSIGNED_BYTE, renderTextureData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

void preparePBO()
{
    glGenBuffers(1, &pboId);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);                                                            // Make this the current UNPACK buffer (OpenGL is state-based)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, CellGridDimension * CellGridDimension * 4, NULL, GL_DYNAMIC_COPY); // Allocate data for the buffer. 4-channel 8-bit image
}

void initCUDAtex()
{
    CUDA_CALL(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pboId, cudaGraphicsRegisterFlagsWriteDiscard));
}

void releaseOpenGL()
{
    if (textureId > 0)
        glDeleteTextures(1, &textureId);
    if (pboId > 0)
        glDeleteBuffers(1, &pboId);
}
void releaseCUDA()
{
    cudaGraphicsUnregisterResource(cudaPBOResource);
    cudaGraphicsUnregisterResource(cudaTexResource);
}

void releaseResources()
{

    releaseCUDA();
    releaseOpenGL();

    averageEvolveTime /= (double)iter;
    averageFitnessTime /= (double)iter;

    printf("Average evolve time: %f ms\n", averageEvolveTime);
    printf("Average fitness time: %f ms\n", averageFitnessTime);
}
#endif
int main(int argc, char **argv)
{
    std::string inputFile = "/home/mor0146/github/CudaCourse/project/images/radial16bit_2.png";
    if (argc > 1)
    {
        inputFile = argv[1];
    }
#if ENABLE_OGL
    // Initialize CellGrid
    fprintf(stdout, "Cell count: %lu\n", CellCount);
    KernelSettings ks = {};
    ks.blockDimension = dim3(ThreadsPerBlock, ThreadsPerBlock, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, ThreadsPerBlock), get_number_of_parts(CellGridDimension, ThreadsPerBlock), 1);
    grid = CellGrid(CellGridDimension, CellGridDimension, ks);

    //Image fitnessImage = Image("/home/mor0146/github/CudaCourse/project/images/radial16bit_2.png", ImageType_GrayScale_16bpp);
    fprintf(stdout, "Loading %s as fitness image.\n", inputFile.c_str());
    Image fitnessImage = Image(inputFile.c_str(), ImageType_GrayScale_16bpp);
    grid.initialize_grid(fitnessImage);

    // Start OpenGL
    initGL(argc, argv);

    create_render_texture();

    preparePBO();

    initCUDAtex();

    //start rendering mainloop
    glutMainLoop();
    atexit(releaseResources);

    free(renderTextureData);

    fprintf(stdout, "terminated\n");
    return 0;
#else
    fprintf(stdout, "Cell count: %lu\n", CellCount);
    KernelSettings ks = {};
    ks.blockDimension = dim3(ThreadsPerBlock, ThreadsPerBlock, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, ThreadsPerBlock), get_number_of_parts(CellGridDimension, ThreadsPerBlock), 1);
    CellGrid grid = CellGrid(CellGridDimension, CellGridDimension, ks);

    Image fitnessImage = Image("/home/mor0146/github/CudaCourse/project/images/radial16bit_2.png", ImageType_GrayScale_16bpp);
    grid.initialize_grid(fitnessImage);

    float fitness = 0.0;
    float lastFitness = -1.0f;
    uint iter = 0;
    double diff = 0;
    double averageEvolveTime = 0.0f;
    double averageFitnessTime = 0.0f;
    size_t sameFitnessValue = 0;
    while (iter < NumberOfEvolutions && sameFitnessValue < 5)
    {
        float evolveTime, fitnessTime;

        lastFitness = fitness;
        ++iter;

        grid.evolve(evolveTime);
        averageEvolveTime += evolveTime;

        fitness = grid.get_average_fitness(fitnessTime);
        averageFitnessTime += fitnessTime;

        diff = fitness - lastFitness;

        if (fitness == lastFitness)
            sameFitnessValue++;
        else
            sameFitnessValue = 0;

        printf("Finished iteration %u, fitness: %.6f\t %.6f\n", iter + 1, fitness, diff); //diff >= 0 ? "+" : "-",
    }

    averageEvolveTime /= (double)iter;
    averageFitnessTime /= (double)iter;

    printf("Average evolve time: %f ms\n", averageEvolveTime);
    printf("Average fitness time: %f ms\n", averageFitnessTime);
#endif
}

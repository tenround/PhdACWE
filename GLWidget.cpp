#include <QtGui/QMouseEvent>
#include <QFileInfo>
#include <glew.h>
#include <fstream>

#include <GL/gl.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "GLWidget.h"
#include "GLManager/GLManager.h"
#include "FileManager/FileManager.h"
#include "debug.h"

#define ARRAY_COUNT( array ) (sizeof( array ) / (sizeof( array[0] ) * (sizeof( array ) != sizeof(void*) || sizeof( array[0] ) <= sizeof(void*))))

// Couple of colors
#define RED     1.0f, 0.0f, 0.0f, 1.0f
#define GREEN   0.0f, 1.0f, 0.0f, 1.0f
#define BLUE    0.0f, 0.0f, 1.0f, 1.0f
#define YELLOW  1.0f, 1.0f, 0.0f, 1.0f
#define WHITE   1.0f, 1.0f, 1.0f, 1.0f

#define NUM_SAMPLERS = 1;

//Vertex colors are then defined only by vertex from 0 to 7 
float vertexColors[] = {
    0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 1.0f
};


float textCoords[] = {
    0.0f, 1.0f, //0
    1.0f, 1.0f, //1
    1.0f, 0.0f, //3
    0.0f, 0.0f, //2
};

const float size = .9;
const float zval = 1;//Bigger the number farther away

//Positions of a square
float vertexPositions[] = {
    -size, size, zval, 1.0f,//Upper left
    size, size, zval, 1.0f, //Upper right
    size, -size, zval, 1.0f, // Lower right
    -size, -size, zval, 1.0f //Lower left
};

//Indexes of the elements of an array
unsigned int vertexIndexes[] = {0, 1, 2, 3};

using namespace std;

GLWidget::GLWidget(QWidget *parent) : QGLWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);

    z = 1; // Default depth to show the pictures
    hsize = .9; // Default size for the 'square' to show the picture

    modelMatrix = glm::mat4x4(1.0f);

    //Offset for the objects, in this case is only one object without offset
    offsets[0] = glm::vec3(0.0f, 0.0f, 0.0f);

    tbo_in = 0; //Texture buffer object
    tbo_out = 0; //Texture buffer object
    sampler = 0;
    textUnit = 0;

    maxActCountIter = 1200;
    currIter = 0;
    iterStep = 5;
    acIterate = false;
    //Cool: 12, 13, 2, 6
    acExample = 1; //Example 7 is 128x128
    useAllBands = true;

    mask = new int[4];
}

/**
 * This initializes the mask, TODO make it by the user 
 * @param example
 */
void GLWidget::InitMaskSetExample(int example) {
    switch (example) {
        case 0:
            mask[0] = 3;
            mask[1] = 4;
            mask[2] = 7;
            mask[3] = 8;
            inputImage = (char*) "images/Mini.png";
            outputImage = (char*) "images/MiniResult.png";
            break;
        case 1:
            mask[0] = 350;
            mask[1] = 450;
            mask[2] = 750;
            mask[3] = 850;
            inputImage = (char*) "images/RectTest1.png";
            outputImage = (char*) "images/RectTestResult.png";
            break;
        case 2:
            mask[0] = 800;
            mask[1] = 900;
            mask[2] = 250;
            mask[3] = 350;
            inputImage = (char*) "images/SmallRealTest.png";
            outputImage = (char*) "images/SmallRealTestResult.png";
            break;
        case 3:
            mask[0] = 150;
            mask[1] = 250;
            mask[2] = 80;
            mask[3] = 150;
            inputImage = (char*) "images/airplane.jpg";
            outputImage = (char*) "images/airplaneResult.png";
            break;
        case 4:
            mask[0] = 350;
            mask[1] = 450;
            mask[2] = 750;
            mask[3] = 850;
            inputImage = (char*) "images/RectTest1Vect.png";
            outputImage = (char*) "images/RectTestVectResult.png";
            break;
        case 5:
            mask[0] = 70;
            mask[1] = 100;
            mask[2] = 70;
            mask[3] = 100;
            inputImage = (char*) "images/TestSizes/128.png";
            outputImage = (char*) "images/TestSizes/128Result.png";
            break;
        case 6:
            mask[0] = 140;
            mask[1] = 200;
            mask[2] = 140;
            mask[3] = 200;
            inputImage = (char*) "images/TestSizes/256.png";
            outputImage = (char*) "images/TestSizes/256Result.png";
            break;
        case 7:
            mask[0] = 280;
            mask[1] = 400;
            mask[2] = 280;
            mask[3] = 400;
            inputImage = (char*) "images/TestSizes/512.png";
            outputImage = (char*) "images/TestSizes/512Result.png";
            break;
        case 8:
            mask[0] = 560;
            mask[1] = 800;
            mask[2] = 560;
            mask[3] = 800;
            inputImage = (char*) "images/TestSizes/1024.png";
            outputImage = (char*) "images/TestSizes/1024Result.png";
            break;
        case 9:
            mask[0] = 1120;
            mask[1] = 1600;
            mask[2] = 1120;
            mask[3] = 1600;
            inputImage = (char*) "images/TestSizes/2048.png";
            outputImage = (char*) "images/TestSizes/2048Result.png";
            break;
        case 10:
            mask[0] = 400;
            mask[1] = 600;
            mask[2] = 300;
            mask[3] = 500;
            inputImage = (char*) "images/planet1.jpg";
            outputImage = (char*) "images/planet1Result.png";
            break;
        case 11:
            mask[0] = 400;
            mask[1] = 600;
            mask[2] = 300;
            mask[3] = 500;
            inputImage = (char*) "images/planet2.jpg";
            outputImage = (char*) "images/planet2Result.png";
            break;
        case 12:
            //Width
            mask[0] = 200;
            mask[1] = 280;
            //Height
            mask[2] = 320;
            mask[3] = 400;
            inputImage = (char*) "images/niiT1.png";
            outputImage = (char*) "images/niiT1Result.png";
            break;
        case 13:
            mask[0] = 200;
            mask[1] = 300;
            mask[2] = 150;
            mask[3] = 250;
            inputImage = (char*) "images/planet3.jpg";
            outputImage = (char*) "images/planet3Result.png";
            break;

    }
}

void GLWidget::CreateSamplers() {
    int num_samplers = 1;
    glGenSamplers(num_samplers, &samplerID[0]);

    for (int samplerIx = 0; samplerIx < num_samplers; samplerIx++) {
        //Defines the Wraping parameter for all the samplers as GL_REPEAT
        glSamplerParameteri(samplerID[samplerIx], GL_TEXTURE_WRAP_S, GL_REPEAT);
        glSamplerParameteri(samplerID[samplerIx], GL_TEXTURE_WRAP_T, GL_REPEAT);
        glSamplerParameteri(samplerID[samplerIx], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(samplerID[samplerIx], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    //    Using GL_LINEAR interpolation for the sampler
    //glSamplerParameteri(samplerID[0], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glSamplerParameteri(samplerID[0], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(samplerID[0], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glSamplerParameteri(samplerID[0], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void GLWidget::InitActiveCountours() {
    int method = 2; //1 for OZ 2 for Voronoi
    float alpha = 0.5;
    float def_dt = .5;

    switch (method) {
        case 1:
            clObj.loadProgram(SDFOZ, (char*) inputImage, (char*) outputImage,
                    maxActCountIter, alpha, def_dt, mask);
            break;
        case 2:
            clObj.loadProgram(SDFVORO, (char*) inputImage, (char*) outputImage,
                    maxActCountIter, alpha, def_dt, mask);
            break;
    }
}

void GLWidget::InitializeVertexBuffer() {
    GLManager::CreateBuffer(vbo_pos, vertexPositions, sizeof (vertexPositions),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_color, vertexColors, sizeof (vertexColors),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 1, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_tcord, textCoords, sizeof (textCoords),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 2, 2, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLWidget::InitTextures() {

    //inputImage = "images/planet1.jpg";

    BYTE* imageTemp = ImageManager::loadImageByte(inputImage, width, height);
    float* image = ImageManager::byteToFloatNorm(imageTemp, width * height * 4);

    dout << "Size of byte: " << sizeof(BYTE) << endl;
    dout << "Size of char: " << sizeof(char) << endl;
    //for(int i=1; i<100; i++){
    //    dout << (float)image[i]*255 << endl;
    //}

    int maxDim = max(width, height);
    //Top left
    vertexPositions[0] = (float) -width / maxDim;
    vertexPositions[1] = (float) height / maxDim;

    //Top Right
    vertexPositions[4] = (float) width / maxDim;
    vertexPositions[5] = (float) height / maxDim;

    //Bottom Right
    vertexPositions[8] = (float) width / maxDim;
    vertexPositions[9] = (float) -height / maxDim;

    //Bottom Left 
    vertexPositions[12] = (float) -width / maxDim;
    vertexPositions[13] = (float) -height / maxDim;

    GLManager::Create2DTexture(tbo_in, image, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);
    GLManager::Create2DTexture(tbo_out, NULL, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);

    delete[] image;
}

void GLWidget::InitializeProgram() {
    std::vector<GLuint> shaderList;

    //Reads the vertex and fragment shaders
    string strVertexShader = FileManager::readFile("VertShader.vp");
    string strFragmentShader = FileManager::readFile("FragShader.fp");

    //dout << "Vertex shader:" << strVertexShader <<endl;
    //dout << "Fragment shader:" << strFragmentShader <<endl;
    shaderList.push_back(GLManager::CreateShader(GL_VERTEX_SHADER, strVertexShader));
    shaderList.push_back(GLManager::CreateShader(GL_FRAGMENT_SHADER, strFragmentShader));

    //Compiles and links the shaders into a program
    g_program.theProgram = GLManager::CreateProgram(shaderList);

    //    LoadShader("VertShader.vp", "FragShader.fp");

    dout << "Program compiled and linked" << endl;
    //Gets the uniform id for the camera to clip martrix (perspective projection)
    g_program.cameraToClipMatrixUnif = glGetUniformLocation(g_program.theProgram, "perspectiveMatrix");

    //Gets the uniform for the model to camera matrix (movement of each object)
    modelToCameraMatrixUnif = glGetUniformLocation(g_program.theProgram, "modelMatrix");
    dout << "MatrixUnif: " << modelToCameraMatrixUnif << endl;

    GLuint textSamplerLoc = glGetUniformLocation(g_program.theProgram, "textSampler");
    dout << "textSamplerLoc: " << textSamplerLoc << endl;

    glUseProgram(g_program.theProgram); //Start using the builded program

    glUniform1i(textSamplerLoc, textUnit); //Binds the texture uniform with the texture like id
    glUniformMatrix4fv(g_program.cameraToClipMatrixUnif, 1, GL_FALSE, glm::value_ptr(camera->getCameraMatrix()));
    glUseProgram(0);

    dout << "--------------End of loading things" << endl;
}

/**
 * Makes all the initialization
 */
void GLWidget::init() {

    Timer tm_oclogl_init(ts, "OCLinit");
    Timer tm_ocl_sdf(ts, "SDF");

    tm_oclogl_init.start();

//Initializes the camera perspective paramteres
    float fzNear = 1.0f;
    float fzFar = 1000.0f;
    float FOV = 45.0f;

    camera = new FPSMovement(fzNear, fzFar, FOV);

    dout << "Initializing program... " << endl;
    InitializeProgram();
    dout << "Program initialized ... " << endl;

    //Create the Vertex Array Object (contains info of vertex, color, coords, and textures)
    glGenVertexArrays(1, &vaoID); //Generate 1 vertex array
    glBindVertexArray(vaoID); //First VAO setup (only one this time)

    dout << "Initializing OpenCL... " << endl;
    InitMaskSetExample(acExample);
    InitActiveCountours();

    dout << "Initializing Textures... " << endl;
    InitTextures(); //Init textures
    dout << "Textures initialized!! " << endl;
    CreateSamplers();

    dout << "Initializing Vertex buffers... " << endl;
    InitializeVertexBuffer(); //Init Vertex buffers

    dout << "Initializing images, arrays and buffers (CL)!! " << endl;
    clObj.initImagesArraysAndBuffers(tbo_in, tbo_out);

    tm_ocl_sdf.start();
    clObj.runSDF();
    tm_ocl_sdf.end();
    tm_oclogl_init.end();

    dout << "Init SUCCESSFUL................" << endl;

    glBindVertexArray(0); //Unbind any vertex array

    clObj.iterate(iterStep, useAllBands); //Iterate the ActiveCountours n times
}

void GLWidget::initializeGL() {
    
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    }

    glEnable(GL_CULL_FACE);//Cull ('desechar') one or more faces of polygons
    glCullFace(GL_BACK);// Hide the 'back' face
    glFrontFace(GL_CW);//Which face is 'front' face, defindes as Clock Wise

    init();
}

void GLWidget::resizeGL(int w, int h) {
    dout << "Resizing GL ......." << endl;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

void GLWidget::paintGL() {
    dout << "Displaying ...." << endl;
    glFlush();

    Timer tm_ocl_ac(ts, "ACont");
    if ((currIter < maxActCountIter) && acIterate) {
		dout << "iterating ....." << endl;
        tm_ocl_ac.start();
        clObj.iterate(iterStep, useAllBands); //Iterate the ActiveCountours n times
        currIter += iterStep;
        tm_ocl_ac.end();
        dout << "Current iter: " << currIter << endl;
        ts.dumpTimings();
    }


    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(g_program.theProgram);

    glBindVertexArray(vaoID); //First VAO setup (only one this time)

	modelMatrix[3] = glm::vec4(offsets[0], 1.0f);
	
    glUniformMatrix4fv(modelToCameraMatrixUnif, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    glActiveTexture(GL_TEXTURE0 + textUnit);
    glBindTexture(GL_TEXTURE_2D, tbo_in);
    glBindSampler(textUnit,samplerID[0]);

    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

    glBindTexture(GL_TEXTURE_2D, tbo_out);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

    //-------- TEXTURES ----------
    glBindSampler(textUnit, 0);
    glBindTexture(GL_TEXTURE_2D,0);

    glBindVertexArray(0);//Unbind VAO
    glUseProgram(0);//Unbind program
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event) {
    dout << "Inside Mouse press event" << endl;
}

void GLWidget::mouseMoveEvent(QMouseEvent *event) {
    printf("%d, %d\n", event->x(), event->y());
}

void GLWidget::keyPressEvent(QKeyEvent* event) {

    cout << "Key= " << (unsigned char)event->key() << endl;
    camera->Keyboard((unsigned char)event->key(), 0, 0);
    //printMatrix(camera->getCameraMatrix());
    switch (event->key()) {

        case 105:// Case 'I' start and stops Active Contours
        case 73:
            acIterate = !acIterate;
            break;
        case 66:// Case 'B' toggle using all bands or only red band
        case 98:
            useAllBands = !useAllBands;
            break;
        case 116:// Case 'T' shows the timings
        case 84:
            ts.dumpTimings();
            break;
        case Qt::Key_Escape:
            close();
            break;
        default:
            event->ignore();
            break;

    }

    QWidget::keyPressEvent(event);

    updateGL();
    //UpdatePerspective();
    //glutPostRedisplay();
}

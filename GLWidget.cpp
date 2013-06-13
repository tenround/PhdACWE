#include <QtGui/QMouseEvent>
#include <QFileInfo>
#include <QFileDialog>
#include <glew.h>
#include <fstream>

#include <GL/gl.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>

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
const float zval = 1; //Bigger the number farther away

//Positions of a square that the user
// is drawing
float vertexPosSelection[] = {
    -0, 0, zval-.01, 1.0f, //Upper left
    0, 0, zval-.01, 1.0f, //Upper right
    0, -0, zval-.01, 1.0f, // Lower right
    -0, -0, zval-.01, 1.0f //Lower left
};


//Positions of a square
float vertexPositions[] = {
    -size, size, zval, 1.0f, //Upper left
    size, size, zval, 1.0f, //Upper right
    size, -size, zval, 1.0f, // Lower right
    -size, -size, zval, 1.0f //Lower left
};

//Indexes of the elements of an array
unsigned int vertexIndexes[] = {0, 1, 2, 3};

using namespace std;

/**
 * Constructor of the Widget. It sets the default
 * values of all its internal properties.
 */
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

    maxActCountIter = 12000;
    currIter = 0;
    iterStep = 5;
    acIterate = false;
    //Cool: 12, 13, 2, 6
    acExample = 1; //Example 7 is 128x128
    useAllBands = true;

    mask = new int[4];

    imageSelected = false;
    newMask = false;
    displaySegmentation = false;

    //	SelectImage();
}

/**
 * Opens the 'select image' dialog. Stops all previous
 * segmentation and reloads everything.
 * TODO use exceptions or something similar to avoid returning ints
 */
int GLWidget::SelectImage() {
    //QString fileName = QFileDialog::getOpenFileName(this, tr("Select an image"), "", tr("Files (*.*)"));
    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/RectTest1.png";
    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/Ft.png";
    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/Min.png";
    QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/planet1.jpg";

    if (!fileName.isNull()) {
        inputImage = new char[fileName.length() + 1];
        outputImage = new char[fileName.length() + 9];

        strcpy(inputImage, fileName.toLatin1().constData());
        dout << "Input image: " << inputImage << endl;

        fileName = fileName.replace(QString("."), QString("_result."));
        strcpy(outputImage, fileName.toLatin1().constData());
        dout << "Output image: " << outputImage << endl;

        //Clear selection of mask
        updatingROI = false;
        startXmask = -1;
        startYmask = -1;
        endXmask = -1;
        endYmask = -1;

        //When we select a new image we stop showing
        // the 'segmentation' until a new ROI is selected
        // TODO clear ROI
        displaySegmentation = false;
        return 1;
    } else {
        //TODO display a dialog informing the following text.
        cout << "The image haven't been selected. " << endl;
        return 0;
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
    cout << "InitActiveCountours" << endl;
    int method = 2; //1 for OZ 2 for Voronoi
    float alpha = 0.5;
    float def_dt = .5;

    cout << "Delete. At InitActiveCountours size is: (" << width << "," << height << ")" << endl;
    switch (method) {
        case 1:
            clObj.loadProgram(SDFOZ, (char*) inputImage, (char*) outputImage,
                    maxActCountIter, alpha, def_dt, width, height);
            break;
        case 2://Current
            clObj.loadProgram(SDFVORO, (char*) inputImage, (char*) outputImage,
                    maxActCountIter, alpha, def_dt, width, height);
            break;
    }
}

void GLWidget::InitializeSimpleVertexBuffer() {

    GLManager::CreateBuffer(vbo_selection, vertexPosSelection, sizeof (vertexPosSelection),
            GL_ARRAY_BUFFER, GL_STREAM_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);
}
/**
 * This function initializes the vertex positions. In this
 * case we simply have a big square that has the size of the window
 */
void GLWidget::InitializeVertexBuffer() {

    dout << "At InitializeVertexBuffer size is: (" << width << "," << height << ")" << endl;
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

    GLManager::CreateBuffer(vbo_pos, vertexPositions, sizeof (vertexPositions),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_color, vertexColors, sizeof (vertexColors),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 1, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_tcord, textCoords, sizeof (textCoords),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 2, 2, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);

    //Unbind buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLWidget::InitTextures() {

    //inputImage = "images/planet1.jpg";
    //inputImage = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/planet1.jpg";
    BYTE* imageTemp = ImageManager::loadImageByte(inputImage, width, height);
    float* image = ImageManager::byteToFloatNorm(imageTemp, width * height * 4);

    dout << "Size of byte: " << sizeof (BYTE) << endl;
    dout << "Size of char: " << sizeof (char) << endl;

    //cout << "Delete. At InitTextures size is: (" << width << "," << height << ")" << endl;

    //for(int i=1; i<100; i++){
    //    dout << (float)image[i]*255 << endl;
    //}

    GLManager::Create2DTexture(tbo_in, image, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);
    GLManager::Create2DTexture(tbo_out, NULL, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);

    //float div = 3;

    delete[] image;
}

/**
 * Initializes the shaders for OpenGL. It also
 * initializes the OpenGL program, the camera and the 
 * uniforms */
void GLWidget::InitializeProgram() {
    std::vector<GLuint> shaderList;

    //Reads the vertex and fragment shaders
    string strVertexShader = FileManager::readFile("VertShader.glsl");
    string strFragmentShader = FileManager::readFile("FragShader.glsl");

    //dout << "Vertex shader:" << strVertexShader <<endl;
    //dout << "Fragment shader:" << strFragmentShader <<endl;
    shaderList.push_back(GLManager::CreateShader(GL_VERTEX_SHADER, strVertexShader));
    shaderList.push_back(GLManager::CreateShader(GL_FRAGMENT_SHADER, strFragmentShader));

    //Compiles and links the shaders into a program
    g_program.theProgram = GLManager::CreateProgram(shaderList);

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

    std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);

    dout << "-------- Compiling Simple Fragment shader program ----------" << endl;

    //Reads the vertex and fragment shaders
    string strSimpleFragmentShader = FileManager::readFile("SimpleFragShader.glsl");

    shaderList.push_back(GLManager::CreateShader(GL_VERTEX_SHADER, strVertexShader));
    shaderList.push_back(GLManager::CreateShader(GL_FRAGMENT_SHADER, strSimpleFragmentShader));

    g_program.simpleFragProgram = GLManager::CreateProgram(shaderList);

    dout << "Simpler Program compiled and linked" << endl;
    dout << "--------------End of loading things" << endl;
}

/**
 * Initializes the vertex and textures once the image
 * has been loaded properly. 
 */
void GLWidget::init() {

    //Create the Vertex Array Object (contains info of vertex, color, coords, and textures)
    glGenVertexArrays(1, &vaoID); //Generate 1 vertex array
    glGenVertexArrays(1, &vaoSimpleID); //Generate 1 vertex array
    glBindVertexArray(vaoID); //First VAO setup (only one this time)

    dout << "Initializing Textures... " << endl;
    InitTextures(); //Init textures
    dout << "Textures initialized!! " << endl;
    CreateSamplers();

    dout << "Initializing Vertex buffers... " << endl;
    InitializeVertexBuffer(); //Init Vertex buffers


    // This should be already after mask 
    dout << "Initializing OpenCL... " << endl;
    InitActiveCountours();

    dout << "Initializing images, arrays and buffers (CL)!! " << endl;
    clObj.initImagesArraysAndBuffers(tbo_in, tbo_out);

    dout << "Init SUCCESSFUL................" << endl;

    glBindVertexArray(vaoSimpleID); //First VAO setup (only one this time)
    InitializeSimpleVertexBuffer();
    dout << "Initializing simple VAO (for ROI)" << endl;

    glBindVertexArray(0); //Unbind any vertex array

    //init() gets called after selecting an image.
    // TODO when everything works, move after pressing s
    imageSelected = true;
}

/* This is the first call after the constructor.
 * This method initializes the state for OpenGL.
 */
void GLWidget::initializeGL() {

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    }

    glEnable(GL_CULL_FACE); //Cull ('desechar') one or more faces of polygons
    glCullFace(GL_BACK); // Hide the 'back' face
    glFrontFace(GL_CW); //Which face is 'front' face, defindes as Clock Wise

    Timer tm_oclogl_init(ts, "OCLinit");
    tm_oclogl_init.start();

    //Initializes the camera perspective paramteres
    float fzNear = 1.0f;
    float fzFar = 1000.0f;
    float FOV = 45.0f;

    camera = new FPSMovement(fzNear, fzFar, FOV);

    dout << "Initializing OpenGL program... " << endl;
    InitializeProgram();
    dout << "OpenGL program initialized ... " << endl;

    tm_oclogl_init.end();
}

void GLWidget::resizeGL(int w, int h) {
    dout << "Resizing GL ......." << endl;

    winWidth = w;
    winHeight = h;

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //TODO this should not be here, we need to call it because the 
    // 'select file' window doesnt' work if it is not called from the constructor
    //    init();
}

/**
 * This is the main OpenGL loop. Here the display of results is made
 */
void GLWidget::paintGL() {
    glFlush();

    //Check if we already have an image selected, if not nothing should be done
    if (imageSelected) {
        //        dout << "Painting........... " << endl;

        if (newMask) {
            cout << "--------------- Initializing mask and making SDF..........." << endl;

            Timer tm_ocl_sdf(ts, "SDF");

            tm_ocl_sdf.start();
            clObj.createRGBAMask(width, height, mask[0], mask[1], mask[2], mask[3]);
            clObj.runSDF();
            tm_ocl_sdf.end();

            newMask = false;
        }

        if ((currIter < maxActCountIter) && acIterate) {

            dout << "iterating ....." << currIter << endl;
            Timer tm_ocl_ac(ts, "ACont");
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
        glBindSampler(textUnit, samplerID[0]);

        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

        if (displaySegmentation) {
            glBindTexture(GL_TEXTURE_2D, tbo_out);
            glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);
        }
        //-------- TEXTURES ----------
        glBindSampler(textUnit, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindVertexArray(0); //Unbind VAO
        glUseProgram(0); //Unbind program

        if(!displaySegmentation){
            glUseProgram(g_program.simpleFragProgram);
            glBindVertexArray(vaoSimpleID);

            //vertexPosSelection[0]-=.01;
            GLManager::CreateBuffer(vbo_selection, vertexPosSelection, sizeof (vertexPosSelection),
                    GL_ARRAY_BUFFER, GL_STREAM_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

            glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0); //Unbind VAO
        glUseProgram(0); //Unbind program

    }
    update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event) {
    endXmask = event->x();
    endYmask = event->y();

    /*
    dout << "Updating mask........ " << endl;
    dout << "Start at: (" << startXmask << "," << startYmask << ")" << endl;
    dout << "Ends at: (" << endXmask << "," << endYmask << ")" << endl;

    dout << "Image size : (" << width << "," << height << ")" << endl;
    dout << "Window size : (" << winWidth << "," << winHeight << ")" << endl;
    */

    mask[0] = (int) ((startXmask * width) / winWidth);
    mask[1] = (int) ((endXmask * width) / winWidth);

    mask[2] = height - (int) ((endYmask * height) / winHeight);
    mask[3] = height - (int) ((startYmask * height) / winHeight);

    dout << "Corresp mask start: (" << mask[0] << "," << mask[2] << ")" << endl;
    dout << "Corresp mask end: (" << mask[1] << "," << mask[3] << ")" << endl;

    newMask = true; //Run SDF (start displaying segmentation) 

    updatingROI = false; //Stop drawing user ROI, start displaying segmentation
}

void GLWidget::mousePressEvent(QMouseEvent *event) {

    dout << "************ INIT POS*************" << endl;
    int currX = event->x();
    int currY = event->y();

    float newX = currX / (float) winWidth;
    float newY = (winHeight - currY) / (float) winHeight;

    startXmask = event->x();
    startYmask = event->y();

    newX = newX*2 - 1;
    newY = newY*2 - 1;

    //------ Initialize ROI all into one point ----
    //Upper left x,y
    vertexPosSelection[0] = newX;
    vertexPosSelection[1] = newY;

    //Upper right x,y
    vertexPosSelection[4] = newX;
    vertexPosSelection[5] = newY;

    //Lower right x,y
    vertexPosSelection[8] = newX;
    vertexPosSelection[9] = newY;

    //Lower left x,y
    vertexPosSelection[12] = newX;
    vertexPosSelection[13] = newY;

    updatingROI = true;

}

/**
 * This function catches the mouse move event. It is used when
 * the user is selecting a ROI. It updates the position of the
 * square to display.
 * @param event
 */
void GLWidget::mouseMoveEvent(QMouseEvent *event) {

    if (updatingROI) {
        int currX = event->x();
        int currY = event->y();

        float newX = currX / (float) winWidth;
        float newY = (winHeight - currY) / (float) winHeight;

        //dout << currX << "/" << winWidth << "....." << currY << "/" << winHeight << endl;

        newX = newX*2 -1;
        newY = newY*2 -1;

        //Upper right x,y
        vertexPosSelection[4] = newX;

        //Lower right x,y
        vertexPosSelection[8] = newX;
        vertexPosSelection[9] = newY;

        //Lower left x,y
        vertexPosSelection[13] = newY;
    }
}

/**
 * Management of all the keyboards pressed.
 */
void GLWidget::keyPressEvent(QKeyEvent* event) {

    dout << "Key = " << (unsigned char) event->key() << endl;
    camera->Keyboard((unsigned char) event->key(), 0, 0);

    int success = 0; //Var used to ensure success of actions

    //printMatrix(camera->getCameraMatrix());
    switch (event->key()) {

        case 105:// Case 'I' start and stops Active Contours
        case 73:
            //Start iterating
            acIterate = !acIterate;

            // After running the SDF for the first time we 
            // start displaying the segmentation.
            displaySegmentation = true;

            break;
        case 66:// Case 'B' toggle using all bands or only red band
        case 98:
            useAllBands = !useAllBands;
            break;
        case 116:// Case 'T' shows the timings
        case 84:
            ts.dumpTimings();
            break;
        case 'S':
            success = SelectImage();
            if (success) {
                init();
            }
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

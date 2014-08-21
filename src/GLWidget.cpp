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


const float size = 1;
const float zval = 0; //Smaller or negative the number farther away
const float zvalROI = 0.01; //Slightly in front of the default billboard

//Positions of a square

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

    //Offset for the objects, in this case is only one object without offset
	// it is the 'rectangle' holding the image
    offsets[0] = glm::vec3(0.0f, 0.0f, 0.0f);

    tbo_in = 0; //Texture buffer object
    tbo_out = 0; //Texture buffer object
    sampler = 0;
    textUnit = 0;

    maxActCountIter = 12000;// Maximum number of ACWE iterations
    currIter = 0; // Current ACWE iteration
    iterStep = 5; //Number of ACWE iterations before retrieving result back to CPU
    acIterate = false;
    //Cool: 12, 13, 2, 6
    acExample = 1; //Example 7 is 128x128
    useAllBands = true; // Use all bands as an average for the ACWE algorithm

    mask = new int[4];

    imageSelected = false;//Indicates if the image has already been selected
    newMask = false;
    displaySegmentation = false;

	firstTimeImageSelected = true;

    //	SelectImage();
}// QGLWidget constructor

/**
 * Opens the 'select image' dialog. Stops all previous
 * segmentation and reloads everything.
 * TODO use exceptions or something similar to avoid returning ints
 */
void GLWidget::SelectImage() {
//    QString fileName = QFileDialog::getOpenFileName(this, tr("Select an image"), "/home", tr("Files (*.png *.jpg *.bmp)"));

    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/RectTest1.png";
    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/Ft.png";
    //QString fileName = "/media/USBSimpleDrive/Olmo/OpenCL_Examples/OZ_OpenCL/ActiveCountoursImg/images/Min.png";
//	QString fileName = "./images/planet1.jpg";
	QString fileName;
	
	 QFileDialog dialog(this);
	 dialog.setFileMode(QFileDialog::AnyFile);
	 dialog.setViewMode(QFileDialog::List);
	 QStringList fileNames;
	 if (dialog.exec()){
		 fileNames = dialog.selectedFiles();
		 fileName = fileNames[0];
	 }

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
		init();
		firstTimeImageSelected = false;
    } else {
        //TODO display a dialog informing the following text.
        cout << "The image haven't been selected. " << endl;
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

	if(firstTimeImageSelected){
		switch (method) {
			case 1:
				clObj.loadProgram(SDFOZ, maxActCountIter, alpha, def_dt );
				break;
			case 2://Current
				clObj.loadProgram(SDFVORO, maxActCountIter, alpha, def_dt);
				break;
		}
	}

	clObj.loadImage( (char*) inputImage, (char*) outputImage,
						 width, height);
}

void GLWidget::InitializeSimpleVertexBuffer() {

	float size = 0;//Starts empty
	float zval = 0.1;
	//Initializes the ROI with all 0's
	vertexPosSelection= glm::mat4(0.0f);

	vertexPosSelection[0] = glm::vec4(-size, size, zval, 1.0f);
	vertexPosSelection[1] = glm::vec4(size, size, zval, 1.0f);
	vertexPosSelection[2] = glm::vec4(size, -size, zval, 1.0f);
	vertexPosSelection[3] = glm::vec4(-size, -size, zval, 1.0f);

	printGLMmatrix(vertexPosSelection);
	
    GLManager::CreateBuffer(vbo_selection, glm::value_ptr(vertexPosSelection), sizeof(vertexPosSelection),
            GL_ARRAY_BUFFER, GL_STREAM_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);
}

/**
 * This method should clean any buffer (. 
 */
void GLWidget::DeleteBuffers(){
	glDeleteBuffers(1,&vbo_posX);
	glDeleteBuffers(1,&vbo_posY);
	glDeleteBuffers(1,&vbo_posZ);
	glDeleteBuffers(1,&vbo_color);
	glDeleteBuffers(1,&vbo_tcord);
	glDeleteBuffers(1,&ebo);
}

/**
 * This function initializes the vertex positions. In this
 * case we simply have a big square that has the size of the window
 */
void GLWidget::InitializeVertexBufferX() {

	float size = 1;
	float zval = 0;
	vertexPlaneX= glm::mat4(0.0f);
	vertexPlaneX[0] = glm::vec4(-size, size, zval, 1.0f);
	vertexPlaneX[1] = glm::vec4(size, size, zval, 1.0f);
	vertexPlaneX[2] = glm::vec4(size, -size, zval, 1.0f);
	vertexPlaneX[3] = glm::vec4(-size, -size, zval, 1.0f);

    GLManager::CreateBuffer(vbo_posX, glm::value_ptr(vertexPlaneX), sizeof (vertexPlaneX),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_color, vertexColors, sizeof (vertexColors),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 1, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_tcord, textCoords, sizeof (textCoords),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 2, 2, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);

    //Unbind buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void GLWidget::InitializeVertexBufferY() {

	float size = 1;
	vertexPlaneY= glm::mat4(0.0f);
	vertexPlaneY[0] = glm::vec4(0.0f, size,-size, 1.0f);
	vertexPlaneY[1] = glm::vec4(0.0f, size, size, 1.0f);
	vertexPlaneY[2] = glm::vec4(0.0f,-size, size, 1.0f);
	vertexPlaneY[3] = glm::vec4(0.0f,-size,-size, 1.0f);

    GLManager::CreateBuffer(vbo_posY, glm::value_ptr(vertexPlaneY), sizeof (vertexPlaneY),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_color, vertexColors, sizeof (vertexColors),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 1, 4, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateBuffer(vbo_tcord, textCoords, sizeof (textCoords),
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, 2, 2, GL_FALSE, 0, 0, GL_FLOAT);

    GLManager::CreateElementBuffer(ebo, vertexIndexes, sizeof (vertexIndexes), GL_STATIC_DRAW);

    //Unbind buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void GLWidget::InitializeVertexBufferZ() {

	float size = 1;
	vertexPlaneZ= glm::mat4(0.0f);
	vertexPlaneZ[0] = glm::vec4(-size, 0.0f,-size , 1.0f);
	vertexPlaneZ[1] = glm::vec4( size, 0.0f,-size, 1.0f);
	vertexPlaneZ[2] = glm::vec4( size, 0.0f, size, 1.0f);
	vertexPlaneZ[3] = glm::vec4(-size, 0.0f, size, 1.0f);

    GLManager::CreateBuffer(vbo_posZ, glm::value_ptr(vertexPlaneZ), sizeof (vertexPlaneZ),
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

    BYTE* image = ImageManager::loadImageByte(inputImage, width, height);

    //Warning!!! imageFloat is now only used for displaying the internal values not for OpenGL
    //float* imageFloat = ImageManager::byteToFloat(imageByte, width * height * 4);
    
    //ImageManager::printImageBGRA(width, height, imageFloat);

    dout << "Size of byte: " << sizeof (BYTE) << endl;
    dout << "Size of char: " << sizeof (char) << endl;

    GLManager::Create2DTexture(tbo_in, image, width, height, GL_UNSIGNED_INT_8_8_8_8_REV, GL_RGBA, GL_LINEAR, GL_LINEAR);
    //GLManager::Create2DTexture(tbo_out, NULL, width, height, GL_UNSIGNED_INT_8_8_8_8_REV, GL_RGBA16, GL_LINEAR, GL_LINEAR);
    
    //GLManager::Create2DTexture(tbo_in, image, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);
    GLManager::Create2DTexture(tbo_out, NULL, width, height, GL_FLOAT, GL_RGBA16, GL_LINEAR, GL_LINEAR);
}

void GLWidget::printGLVersions() {

	/*
	 	GLint nExtensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);

	for(int i=0; i<nExtensions; i++){ cout << glGetStringi(GL_EXTENSIONS, i) << endl; }
*/
	const GLubyte *renderer = glGetString( GL_RENDER);
	const GLubyte *vendor= glGetString( GL_VENDOR);
	const GLubyte *version = glGetString( GL_VERSION);
	const GLubyte *glsVersion= glGetString( GL_SHADING_LANGUAGE_VERSION);

	cout << "GL Vendor: " << vendor << endl;
	cout << "GL Version : " << version << endl;
	cout << "GLS version: " << glsVersion << endl;
	
}


/**
 * Initializes the shaders for OpenGL. It also
 * initializes the OpenGL program, the camera and the 
 * uniforms */
void GLWidget::InitShaders() {

	printGLVersions();
	
    std::vector<GLuint> shaderList;

    //Reads the vertex and fragment shaders
    string strVertexShader = FileManager::readFile("src/resources/shaders/VertShader.glsl");
    string strFragmentShader = FileManager::readFile("src/resources/shaders/FragShader.glsl");

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
    glUniformMatrix4fv(g_program.cameraToClipMatrixUnif, 1, GL_FALSE, 
            glm::value_ptr(camera->getProjectionMatrix() * camera->getViewMatrix()));
    glUseProgram(0);

    std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);

    dout << "-------- Compiling Simple Fragment shader program ----------" << endl;

    //Reads the vertex and fragment shaders
    string strSimpleFragmentShader = FileManager::readFile("src/resources/shaders/SimpleFragShader.glsl");

    shaderList.push_back(GLManager::CreateShader(GL_VERTEX_SHADER, strVertexShader));
    shaderList.push_back(GLManager::CreateShader(GL_FRAGMENT_SHADER, strSimpleFragmentShader));

	//------------- For lighting ---------
    normalHandle = glGetUniformLocation(normalUnif, "vertexNormal");

    g_program.simpleFragProgram = GLManager::CreateProgram(shaderList);


    dout << "Simpler Program compiled and linked" << endl;
    dout << "--------------End of loading OpenGL Shaders -----------------" << endl;
}

/**
 * Initializes the vertex and textures once the image
 * has been loaded properly. 
 */
void GLWidget::init() {
	dout << "------- init()--------" << endl;

    //IMPORTANT!!!! The textures need to be initialized before the vertex buffers,
    //because it is in this function where the size of the images get read
    dout << "Initializing Textures... " << endl;
    InitTextures(); //Init textures
    dout << "Textures initialized!! " << endl;

	if(firstTimeImageSelected){
		//Create the Vertex Array Object (contains info of vertex, color, coords, and textures)
		glGenVertexArrays(1, &vaoIdX); //Generate 1 vertex array
		glGenVertexArrays(1, &vaoIdY); //Generate 1 vertex array
		glGenVertexArrays(1, &vaoIdZ); //Generate 1 vertex array
		glGenVertexArrays(1, &vaoSimpleID); //Generate 1 vertex array

		// Samplers that define how to treat the image on the corners,
		// and when we zoom in or out to the image
		CreateSamplers();

		dout << "Initializing Vertex buffers... " << endl;
		glBindVertexArray(vaoIdX); //First VAO setup (only one this time)
		InitializeVertexBufferX(); //Init Vertex buffers
		glBindVertexArray(vaoIdY); 
		InitializeVertexBufferY(); 
		glBindVertexArray(vaoIdZ); 
		InitializeVertexBufferZ(); 
	}

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

    glDisable(GL_CULL_FACE); //Cull ('desechar') one or more faces of polygons
//    glCullFace(GL_BACK); // Hide the 'back' face
//    glFrontFace(GL_CW); //Which face is 'front' face, defines as Clock Wise
	glEnable(GL_DEPTH_TEST);//Enables depth buffer
	glDepthFunc(GL_LEQUAL);//Indicates the depth function to use

    Timer tm_oclogl_init(ts, "OCLinit");
    tm_oclogl_init.start();

    //Initializes the camera perspective paramteres
    float fzNear = 0.1f;
    float fzFar = 1000.0f;
    float FOV = 45.0f;

    camera = new FPSMovement(fzNear, fzFar, FOV);

    dout << "Initializing OpenGL program... " << endl;
    InitShaders();
    dout << "OpenGL program initialized ... " << endl;

    tm_oclogl_init.end();
}

void GLWidget::resizeGL(int w, int h) {
    dout << "Resizing GL ......." << endl;

    // NEVER TOUCH THIS TWO VALUES ARE NECESSARY
    winWidth = w;//Updating the width of the window for the ROI
    winHeight = h;//Updating the height of the window for the ROI

    camera->Reshape(w,h);
    glUniformMatrix4fv(g_program.cameraToClipMatrixUnif, 1, GL_FALSE, 
            glm::value_ptr(camera->getProjectionMatrix() * camera->getViewMatrix()));

}

/**
 * This is the main OpenGL loop. Here the display of results is made
 */
void GLWidget::paintGL() {
    glFlush();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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


        modelMatrix = camera->getModelMatrix();

        glUniformMatrix4fv(modelToCameraMatrixUnif, 1, 
                GL_FALSE, glm::value_ptr(modelMatrix));

        glUniformMatrix4fv(g_program.cameraToClipMatrixUnif, 1, GL_FALSE, 
            glm::value_ptr(camera->getProjectionMatrix() * camera->getViewMatrix()));

        glActiveTexture(GL_TEXTURE0 + textUnit);
        glBindTexture(GL_TEXTURE_2D, tbo_in);
        glBindSampler(textUnit, samplerID[0]);

        glBindVertexArray(vaoIdX); //First VAO setup (only one this time)
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

        glBindVertexArray(vaoIdY); 
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

        glBindVertexArray(vaoIdZ); 
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

		//This is displaying the results of the segmentation (from a texture
		// computed suing OpenCL)
        if (displaySegmentation) {
            glBindTexture(GL_TEXTURE_2D, tbo_out);
            glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);
        }
        //-------- TEXTURES ----------
        glBindSampler(textUnit, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindVertexArray(0); //Unbind VAO
        glUseProgram(0); //Unbind program

		// This is used to display the ROI that the user is selecting
        if(!displaySegmentation){
            glUseProgram(g_program.simpleFragProgram);
            glBindVertexArray(vaoSimpleID);//Attach the VAO for displaying ROI

			GLManager::CreateBuffer(vbo_selection, glm::value_ptr(vertexPosSelection), sizeof(vertexPosSelection),
					GL_ARRAY_BUFFER, GL_STREAM_DRAW, 0, 4, GL_FALSE, 0, 0, GL_FLOAT);

            glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0); //Unbind VAO
        glUseProgram(0); //Unbind program

    }
    update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event) {
    camera->mouseReleaseEvent(event);

	if( event->button() == GLUT_RIGHT_BUTTON){
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
}

void GLWidget::mousePressEvent(QMouseEvent *event) {

	cout << "Button: " << event->button() << endl;
    camera->mousePressEvent(event);

    //static int PRIMARY = GLUT_LEFT_BUTTON;//Which mouse button will be used for movement
	if( event->button() == GLUT_RIGHT_BUTTON){

		dout << "************ INIT ROI POS*************" << endl;

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
		vertexPosSelection[0].x = newX;
		vertexPosSelection[0].y = newY;

		//Upper right x,y
		vertexPosSelection[1].x = newX;
		vertexPosSelection[1].y = newY;

		//Lower right x,y
		vertexPosSelection[2].x = newX;
		vertexPosSelection[2].y = newY;

		//Lower left x,y
		vertexPosSelection[3].x = newX;
		vertexPosSelection[3].y = newY;
		updatingROI = true;
	}
}

/**
 * Controls the events when the wheel of the mouse is pressed. 
 * In this case it zooms in and out translating the view matrix 
 * @param event
 */
void GLWidget::wheelEvent(QWheelEvent *event) {

    camera->wheelEvent(event);

	modelMatrix = camera->getModelMatrix();
}
/**
 * This function catches the mouse move event. It is used when
 * the user is selecting a ROI. It updates the position of the
 * square to display.
 * @param event
 */
void GLWidget::mouseMoveEvent(QMouseEvent *event) {

    camera->mouseMoveEvent(event);
    if (updatingROI) {
        int currX = event->x();
        int currY = event->y();

        float newX = currX / (float) winWidth;
        float newY = (winHeight - currY) / (float) winHeight;

//        dout << currX << "/" << winWidth << "....." << currY << "/" << winHeight << endl;

        newX = newX*2 -1;
        newY = newY*2 -1;

        //Upper right x,y
        vertexPosSelection[1].x = newX;

        //Lower right x,y
        vertexPosSelection[2].x = newX;
        vertexPosSelection[2].y = newY;

        //Lower left x,y
        vertexPosSelection[3].y = newY;
    }
}

void GLWidget::keyReleaseEvent(QKeyEvent* event) {
    camera->keyReleaseEvent(event);
}
/**
 * Management of all the keyboards pressed.
 */
void GLWidget::keyPressEvent(QKeyEvent* event) {

    camera->keyPressEvent(event);
    dout << "Key = " << (unsigned char) event->key() << endl;

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
            SelectImage();
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

void GLWidget::printGLMmatrix(glm::mat4 matrix)
{
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[0].x, matrix[0].y, matrix[0].z, matrix[0].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[1].x, matrix[1].y, matrix[1].z, matrix[1].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[2].x, matrix[2].y, matrix[2].z, matrix[2].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[3].x, matrix[3].y, matrix[3].z, matrix[3].w);
}


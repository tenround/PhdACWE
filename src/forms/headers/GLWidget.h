#ifndef _GLWIDGET_H
#define _GLWIDGET_H

//glew import HAS TO BE before QGLWidget
#include <GL/glew.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#include "CameraMovement.h"
#include "FPSMovement.h"

#include "GLManager/GLManager.h"
#include "CLManager/CLManager.h"
#include "ImageManager/ImageManager.h"
#include "ActiveContours.h"

#include "Timers/timing.h"
#include <QtOpenGL/QGLWidget>
#include <QGLShaderProgram>
#include <QGLShader>

#include <glm/glm.hpp>

struct ProgramData {
    GLuint theProgram;
    GLuint simpleFragProgram;
    GLuint cameraToClipMatrixUnif;
};

class GLWidget : public QGLWidget {
    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent = NULL);
    void printMatrix(glm::mat4 matrix);
    void SelectImage();

protected:
	void DeleteBuffers();
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void init();
    void initMaskLimits();
    void InitShaders();
    void InitTextures();
    void initImgVaoBuffer();
    void initPlanesVaoBuffer();
    void initTextureCoords();
    void initSegmVaoBuffer();
    void InitActiveCountours();
    void CreateSamplers();
    void printGLMmatrix(glm::mat4 matrix);
	void printGLVersions();
	void InitVertexData();
    void createMaskRunSDF();

	//--------- 3D -------
	void initTexture3D();

private:

	// 3D texturing
	float* img3DText;
	float* result3DText;

	GLuint tbo_in; //Texture buffer object
    GLuint tbo_out; //Texture buffer object
    GLuint imgTextId;
    GLuint segTextId;
    GLuint displaySegmUnif;

    // Used to decide if we display the planes or draw using Ray Casting
    GLuint drawPlanesUnif;
    int drawPlanes;
    GLuint rayCastingDecayUnif;
    float rayCastingDecayValue;

	GLuint errCode; //Texture buffer object

    GLuint modelToCameraMatrixUnif;
    glm::mat4 modelMatrix;

    CameraMovement* camera;

    Timings ts;

    ProgramData g_program;
    glm::vec3 offsets[1];

	float verticesCube[32];
	float verticesPlanes[48];
	glm::mat4 vertexPosSelection;


	float  txcoor[8*3];
	float  txcoorPlanes[12*3];

    //float textCoords[];
    //unsigned int vertexIndexes[];

    float z;
    float hsize;

    GLuint vbo_pos;
    GLuint vbo_pos_planes;
    GLuint vbo_tcords_planes;
    GLuint vbo_tcords;
    GLuint vbo_color;
    GLuint ebo; //Element buffer object
    GLuint ebo_planes; //Element buffer object for planes
    GLuint vbo_selection;

    GLuint vaoId;
    GLuint vaoPlanesId;//Just used to display ROI

    	//Normal of billboard
	GLuint normalUnif; 
	GLuint normalHandle; 

    GLuint samplerID[2];

    //GUI
    bool imageSelected;
    bool newMask;
    bool displaySegmentation;
	bool firstTimeImageSelected; //Indicates if it is the first time an image has been selected

    //Mask selection
    bool updatingROI;
    float startXmask;
    float startYmask;
    float endXmask;
    float endYmask;

    //Window size
    int winWidth;
    int winHeight;

    ActiveContours clObj;
    int maxActCountIter;
    int currIter;
    int iterStep;
    bool acIterate;

    //Cool: 12, 13, 2, 6
    int acExample;
    bool useAllBands;

    //For textures and initial mask

    char* inputImage;
    char* outputImage;
    int* mask;
    //Three positions of the center of the ROI
    int xmask;
    int ymask;
    int zmask;
    //Percentage of the mask size in each dimension. If maskSize = .5 the
    // ROI will be half the size in each direction
    float maskSize;
    // What percentage of the size of the dimension the ROI will be moved 
    // when pressing any of the arrow keys
    float maskMove;
    int width;
    int height;
    int depth;

	float cubeWidth;//This variable represent the with of the cube (always 1)
	float cubeHeight;// Height of the cube. Proportional to width
	float cubeDepth;// Depth of the cube. Proportional to width
	

    QGLShaderProgram *shaderProg;
    QGLShader *vertexShader, *fragmentShader;

};

#endif  /* _GLWIDGET_H */

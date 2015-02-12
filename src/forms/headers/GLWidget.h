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
#include "Timers/timing.h"

#include "ActiveContours.h"
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
    void initMask();
    void InitShaders();
    void InitTextures();
    void initImgVaoBuffer();
    void initTextureCoords();
    void initSegmVaoBuffer();
    void InitActiveCountours();
    void CreateSamplers();
    void printGLMmatrix(glm::mat4 matrix);
	void printGLVersions();
	void InitVertexData();

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

	GLuint errCode; //Texture buffer object

    GLuint modelToCameraMatrixUnif;
    glm::mat4 modelMatrix;

    CameraMovement* camera;

    Timings ts;

    ProgramData g_program;
    glm::vec3 offsets[1];

	float vertexPlanes[32];
	glm::mat4 vertexPosSelection;

	float  txcoor[8*3];
    GLuint vbo_tcords;

    //float textCoords[];
    //unsigned int vertexIndexes[];

    float z;
    float hsize;

    GLuint vbo_pos;
    GLuint vbo_tcord;
    GLuint vbo_color;
    GLuint ebo; //Element buffer object
    GLuint vbo_selection;

    GLuint vaoId;
    GLuint vaoSimpleID;//Just used to display ROI

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

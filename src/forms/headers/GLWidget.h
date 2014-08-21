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
    void InitShaders();
    void InitTextures();
    void InitializeVertexBuffer();
    void InitializeSimpleVertexBuffer();
    void InitActiveCountours();
    void CreateSamplers();
    void printGLMmatrix(glm::mat4 matrix);
	void printGLVersions();

private:
    GLuint modelToCameraMatrixUnif;
    glm::mat4 modelMatrix;

    CameraMovement* camera;

    Timings ts;

    ProgramData g_program;
    glm::vec3 offsets[1];

	glm::mat4 vertexPositions;
	glm::mat4 vertexPosSelection;

    //float textCoords[];
    //unsigned int vertexIndexes[];

    float z;
    float hsize;

    GLuint vbo_pos;
    GLuint vbo_tcord;
    GLuint vbo_color;
    GLuint ebo; //Element buffer object
    GLuint vbo_selection;

    GLuint vaoID;
    GLuint vaoSimpleID;//Just used to display ROI

    GLuint tbo_in; //Texture buffer object
    GLuint tbo_out; //Texture buffer object
    GLuint sampler;
    GLuint textUnit;

	//Normal of billboard
	GLuint normalUnif; 
	GLuint normalHandle; 

    GLuint samplerID[1];

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

    QGLShaderProgram *shaderProg;
    QGLShader *vertexShader, *fragmentShader;

};

#endif  /* _GLWIDGET_H */

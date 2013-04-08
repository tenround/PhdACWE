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

struct ProgramData {
    GLuint theProgram;
    GLuint cameraToClipMatrixUnif;
};

class GLWidget : public QGLWidget {
    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent = NULL);
    void printMatrix(glm::mat4 matrix);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void init();
    void InitializeProgram();
    void InitTextures();
    void InitializeVertexBuffer();
    void InitActiveCountours();
    void CreateSamplers();
    int SelectImage();

private:
    GLuint modelToCameraMatrixUnif;
    glm::mat4 modelMatrix;

    CameraMovement* camera;

    Timings ts;

    ProgramData g_program;
    glm::vec3 offsets[1];

    //float vertexPositions[];
    //float textCoords[];
    //unsigned int vertexIndexes[];

    float z;
    float hsize;

    GLuint vbo_pos;
    GLuint vbo_tcord;
    GLuint vbo_color;
    GLuint ebo; //Element buffer object

    GLuint vaoID;

    GLuint tbo_in; //Texture buffer object
    GLuint tbo_out; //Texture buffer object
    GLuint sampler;
    GLuint textUnit;

    GLuint samplerID[1];

    //GUI
    bool imageSelected;
    bool maskSelected;

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

/* 
 * File:   MainWindow.h
 * Author: olmozavala
 *
 * Created on September 17, 2012, 8:32 AM
 */

#ifndef _MAINWINDOW_H
#define	_MAINWINDOW_H

#include "ui_MainWindow.h"
#include "GLWidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();
    virtual ~MainWindow();
    
protected:
    bool eventFilter(QObject *o, QEvent *e);
    void keyPressEvent(QKeyEvent *event);

private:
    Ui::MainWindow widget;
	GLWidget *glWidget;
};

#endif	/* _MAINWINDOW_H */

import sys
import os
import pandas as pd
from PySide2.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMenuBar, QAction, QFileDialog, QStatusBar, QWidget, QSizePolicy, QLineEdit, QPushButton, QSizePolicy, QMessageBox, QListWidgetItem
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, Qt, QCoreApplication

#导入通量后处理模块
import loaddata
import gapfill
import Ustar_threshold_estimate as ute
import quality_control as qc
import visualization
import partitioning

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("flux proc")
        self.setGeometry(100, 100, 750, 700)
        # 设置最小尺寸
        self.setMinimumSize(600, 600)

        # 创建菜单栏
        self.create_menubar()

        # 创建 QStackedWidget 作为中央小部件
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        # 设置 stacked_widget 的 sizePolicy
        self.stacked_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_menubar(self):
        # 创建菜单栏
        menubar = self.menuBar()

        # 创建菜单项(大类)
        menu_load_data = menubar.addMenu("Load Data")
        menu_quality_control = menubar.addMenu("Quality Control")
        menu_gapfill = menubar.addMenu("GapFill")
        menu_u_threshold = menubar.addMenu("U* Threshold Estimate")
        menu_partitioning = menubar.addMenu("Partitioning")
        menu_visualization = menubar.addMenu("visualization")

        # 添加菜单动作(子类)
        action_load_eddypro = QAction("Load EddyPro", self)
        action_build_dataset = QAction("build dataset", self)
        action_smartY_qc = QAction("smartY_qc", self)
        action_MaxMinValue_qc = QAction("MaxMinValue_qc", self)
        action_Dependency_qc = QAction("Dependency_qc", self)
        action_Datetime_qc = QAction("Datetime_qc", self)
        action_XGboost = QAction("XGboost", self)
        action_random_forest = QAction("Random Forest", self)
        action_Adaboost = QAction("Adaboost", self)
        action_ANN = QAction("ANN", self)
        action_GapFillCols = QAction("Muti-Columns GapFill", self)
        action_MPT = QAction("Moving Point Test(MPT)", self)
        action_plot_fingerprint = QAction("plot columns", self)
        action_plot_scatter = QAction("plot scatter", self)
        action_NT_Reichstein = QAction("NT_Reichstein", self)

        # 连接菜单动作到方法
        action_load_eddypro.triggered.connect(self.show_load_data_page)
        action_build_dataset.triggered.connect(self.show_build_dataset_page)
        action_smartY_qc.triggered.connect(self.show_smartY_qc_page)
        action_MaxMinValue_qc.triggered.connect(self.show_MaxMinValue_qc_page)
        action_Dependency_qc.triggered.connect(self.show_Dependency_qc_page)
        action_Datetime_qc.triggered.connect(self.show_Datetime_qc_page)
        action_XGboost.triggered.connect(self.show_XGboost_page) #lambda: self.load_and_switch_page('XGboost.ui')
        action_random_forest.triggered.connect(self.show_random_forest_page)
        action_Adaboost.triggered.connect(self.show_Adaboost_page) 
        action_ANN.triggered.connect(self.show_ANN_page) 
        action_GapFillCols.triggered.connect(self.show_GapFillCols_page) 
        action_MPT.triggered.connect(self.show_MPT_page) 
        action_plot_fingerprint.triggered.connect(self.show_plot_fingerprint_page) 
        action_plot_scatter.triggered.connect(self.show_plot_scatter_page) 
        action_NT_Reichstein.triggered.connect(self.show_NT_Reichstein_page)

        # 添加动作到相应的菜单
        menu_load_data.addAction(action_load_eddypro)
        menu_load_data.addAction(action_build_dataset)
        menu_quality_control.addAction(action_smartY_qc)
        menu_quality_control.addAction(action_MaxMinValue_qc)
        menu_quality_control.addAction(action_Dependency_qc)
        menu_quality_control.addAction(action_Datetime_qc)
        menu_gapfill.addAction(action_XGboost)
        menu_gapfill.addAction(action_random_forest)
        menu_gapfill.addAction(action_Adaboost)
        menu_gapfill.addAction(action_ANN)
        menu_gapfill.addAction(action_GapFillCols)
        menu_u_threshold.addAction(action_MPT)
        menu_visualization.addAction(action_plot_fingerprint)
        menu_visualization.addAction(action_plot_scatter)
        menu_partitioning.addAction(action_NT_Reichstein)

    def show_load_data_page(self):
        # 创建页面并切换
        load_data_page = load_dataPage()
        self.stacked_widget.addWidget(load_data_page)
        self.stacked_widget.setCurrentWidget(load_data_page)  # 显示页面

    def show_build_dataset_page(self):
        # 创建页面并切换
        build_dataset_page = build_datasetPage()
        self.stacked_widget.addWidget(build_dataset_page)
        self.stacked_widget.setCurrentWidget(build_dataset_page)  # 显示页面

    def show_smartY_qc_page(self):
        # 创建页面并切换
        smartY_qc_page = smartY_qcPage()
        self.stacked_widget.addWidget(smartY_qc_page)
        self.stacked_widget.setCurrentWidget(smartY_qc_page)  # 显示页面

    def show_MaxMinValue_qc_page(self):
        # 创建页面并切换
        MaxMinValue_qc_page = MaxMinValue_qcPage()
        self.stacked_widget.addWidget(MaxMinValue_qc_page)
        self.stacked_widget.setCurrentWidget(MaxMinValue_qc_page)  # 显示页面

    def show_Dependency_qc_page(self):
        # 创建页面并切换
        Dependency_qc_page = Dependency_qcPage()
        self.stacked_widget.addWidget(Dependency_qc_page)
        self.stacked_widget.setCurrentWidget(Dependency_qc_page)  # 显示页面

    def show_Datetime_qc_page(self):
        # 创建页面并切换
        Datetime_qc_page = Datetime_qcPage()
        self.stacked_widget.addWidget(Datetime_qc_page)
        self.stacked_widget.setCurrentWidget(Datetime_qc_page)  # 显示页面
    
    def show_XGboost_page(self):
        # 创建页面并切换
        XGboost_page = XGboostPage()
        self.stacked_widget.addWidget(XGboost_page)
        self.stacked_widget.setCurrentWidget(XGboost_page)  # 显示页面

    def show_random_forest_page(self):
        # 创建页面并切换
        random_forest_page = random_forestPage()
        self.stacked_widget.addWidget(random_forest_page)
        self.stacked_widget.setCurrentWidget(random_forest_page)  # 显示页面

    def show_Adaboost_page(self):
        # 创建页面并切换
        Adaboost_page = AdaboostPage()
        self.stacked_widget.addWidget(Adaboost_page)
        self.stacked_widget.setCurrentWidget(Adaboost_page)  # 显示页面

    def show_ANN_page(self):
        # 创建页面并切换
        ANN_page = ANNPage()
        self.stacked_widget.addWidget(ANN_page)
        self.stacked_widget.setCurrentWidget(ANN_page)  # 显示页面

    def show_GapFillCols_page(self):
        # 创建页面并切换
        GapFillCols_page = GapFillColsPage()
        self.stacked_widget.addWidget(GapFillCols_page)
        self.stacked_widget.setCurrentWidget(GapFillCols_page)  # 显示页面

    def show_MPT_page(self):
        # 创建页面并切换
        MPT_page = MPTPage()
        self.stacked_widget.addWidget(MPT_page)
        self.stacked_widget.setCurrentWidget(MPT_page)  # 显示页面

    def show_plot_fingerprint_page(self):
        # 创建页面并切换
        plot_fingerprint_page = plot_fingerprintPage()
        self.stacked_widget.addWidget(plot_fingerprint_page)
        self.stacked_widget.setCurrentWidget(plot_fingerprint_page)  # 显示页面

    def show_plot_scatter_page(self):
        # 创建页面并切换
        plot_scatter_page = plot_scatterPage()
        self.stacked_widget.addWidget(plot_scatter_page)
        self.stacked_widget.setCurrentWidget(plot_scatter_page)  # 显示页面

    def show_NT_Reichstein_page(self):
        # 创建页面并切换
        NT_Reichstein_page = NT_ReichsteinPage()
        self.stacked_widget.addWidget(NT_Reichstein_page)
        self.stacked_widget.setCurrentWidget(NT_Reichstein_page)  # 显示页面

##-------------------- Load Data --------------------
class load_dataPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'load_data.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.biometButton.clicked.connect(lambda: self.open_file_dialog(self.ui.biometEdit))
        self.ui.fulloutputButton.clicked.connect(lambda: self.open_file_dialog(self.ui.fulloutputEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.runButton.clicked.connect(self.run_load_data)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_load_data(self):
        envidir = self.ui.biometEdit.text()
        fluxdir = self.ui.fulloutputEdit.text()
        outdir = self.ui.outfileEdit.text()

        data = loaddata.LoadData(envidir, fluxdir)
        data.to_csv(outdir, index=False, mode="w")
        QMessageBox.about(self.ui, 'Message','successfully run load_data')

class build_datasetPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'build_dataset.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.ERA5dataButton.clicked.connect(lambda: self.open_file_dialog(self.ui.ERA5dataEdit))
        self.ui.rawdataButton.clicked.connect(lambda: self.open_file_dialog(self.ui.rawdataEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.runButton.clicked.connect(self.run_build_dataset)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_build_dataset(self):
        ERA5datadir = self.ui.ERA5dataEdit.text()
        rawdatadir = self.ui.rawdataEdit.text()
        outdir = self.ui.outfileEdit.text()

        data = gapfill.BuildDataset(ERA5datadir, rawdatadir)
        data.to_csv(outdir, index=False, mode="w")
        QMessageBox.about(self.ui, 'Message','successfully run build_dataset')

##-------------------- Quality Control --------------------
class smartY_qcPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'smartY_qc.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_smartY_qc)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_smartY_qc(self):
        # 检查是否col_to_qc输入框为空
        if not self.ui.col_to_qcEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in col_to_qc.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            col_to_qc_str = self.ui.col_to_qcEdit.text()
            # 将字符串转为列表
            col_to_qc = col_to_qc_str.split(", ")
            col_to_qc = [item.strip("'\"") for item in col_to_qc]# 去除每个元素的单引号和双引号

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = qc.smartY_qc(dataset, col_to_qc)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run smartY_qc')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run smartY_qc: {str(e)}")

class MaxMinValue_qcPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'MaxMinValue_qc.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_MaxMinValue_qc)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_MaxMinValue_qc(self):
        # 检查是否col_to_qc输入框为空
        if not self.ui.col_to_qcEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in col_to_qc.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            col_to_qc_str = self.ui.col_to_qcEdit.text()
            lower_bound_str = self.ui.lower_boundEdit.text() #lower_bound_str示例：'1, 2, 3'
            upper_bound_str = self.ui.upper_boundEdit.text()

            # 将字符串转为列表
            col_to_qc = col_to_qc_str.split(", ")
            col_to_qc = [item.strip("'\"") for item in col_to_qc]# 去除每个元素的单引号和双引号

            if not self.ui.lower_boundEdit.text():
                lower_bound = None
            else:
                # 将字符串转为列表
                lower_bound = [int(item.strip()) for item in lower_bound_str.split(', ')]

            if not self.ui.upper_boundEdit.text():
                upper_bound = None
            else:
                # 将字符串转为列表
                upper_bound = [int(item.strip()) for item in upper_bound_str.split(', ')]

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = qc.MaxMinValue_qc(dataset, col_to_qc, lower_bound=lower_bound, upper_bound=upper_bound)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run MaxMinValue_qc')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run MaxMinValue_qc: {str(e)}")

class Dependency_qcPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'Dependency_qc.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_Dependency_qc)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_Dependency_qc(self):
        # 检查是否col_to_qc输入框为空
        if not self.ui.col_to_qcEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in col_to_qc.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            col_to_qc = self.ui.col_to_qcEdit.text()
            dependency_col_str = self.ui.dependency_colEdit.text()
            lower_bound_str = self.ui.lower_boundEdit.text()
            upper_bound_str = self.ui.upper_boundEdit.text()

            # 将字符串转为列表
            dependency_col = dependency_col_str.split(", ")
            dependency_col = [item.strip("'\"") for item in dependency_col]# 去除每个元素的单引号和双引号

            if not self.ui.lower_boundEdit.text():
                lower_bound = None
            else:
                # 将字符串转为列表
                lower_bound = [int(item.strip()) for item in lower_bound_str.split(', ')]

            if not self.ui.upper_boundEdit.text():
                upper_bound = None
            else:
                # 将字符串转为列表
                upper_bound = [int(item.strip()) for item in upper_bound_str.split(', ')]

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = qc.Dependency_qc(dataset, col_to_qc, dependency_col, lower_bound=lower_bound, upper_bound=upper_bound)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run Dependency_qc')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run Dependency_qc: {str(e)}")

class Datetime_qcPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'Datetime_qc.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        #添加删除日期
        self.ui.addButton.clicked.connect(self.add_time_range)
        self.ui.deleteButton.clicked.connect(self.delete_selected_range)

        #运行程序
        self.ui.startButton.clicked.connect(self.run_Datetime_qc)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)

    def add_time_range(self):
        """添加时间段到列表"""
        start_time = self.ui.startTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end_time = self.ui.endTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss")

        # 检查时间段是否有效
        if self.ui.startTimeEdit.dateTime() >= self.ui.endTimeEdit.dateTime():
            QMessageBox.warning(self, "Invalid Range", "Start time must be earlier than end time!")
            return

        # 添加到列表
        time_range = f"({start_time}, {end_time})"
        self.ui.timeRangeList.addItem(QListWidgetItem(time_range))

    def delete_selected_range(self):
        """删除选中的时间段"""
        selected_items = self.ui.timeRangeList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a time range to delete!")
            return
        for item in selected_items:
            self.ui.timeRangeList.takeItem(self.ui.timeRangeList.row(item))

    def get_time_ranges(self):
        """返回时间段列表"""
        ranges = []
        for index in range(self.ui.timeRangeList.count()):
            item_text = self.ui.timeRangeList.item(index).text()
            start, end = item_text.strip("()").split(", ")
            ranges.append((start, end))
        return ranges
    
    def run_Datetime_qc(self):
        # 检查是否col_to_qc输入框为空
        if not self.ui.col_to_qcEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in col_to_qc.")
        elif self.ui.timeRangeList.count() == 0:
            QMessageBox.warning(self, "No Time Ranges", "Please add at least one time range!")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            col_to_qc = self.ui.col_to_qcEdit.text()
            time_ranges = self.get_time_ranges()  # 获取time_ranges

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = qc.Datetime_qc(dataset, col_to_qc, time_ranges)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run Datetime_qc')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run Datetime_qc: {str(e)}")

##-------------------- Gap Fill --------------------
class XGboostPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'XGboost.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_XGboost_gapfilling)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_XGboost_gapfilling(self):
        # 检查是否有输入框为空
        if not self.ui.ModeEdit.text() or not self.ui.var_to_fillEdit.text() or not self.ui.X_colEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            var_to_fill = self.ui.var_to_fillEdit.text()
            Mode = self.ui.ModeEdit.text()
            X_col_str = self.ui.X_colEdit.text()
            # 将字符串转为列表
            X_col = X_col_str.split(", ")
            X_col = [item.strip("'\"") for item in X_col]# 去除每个元素的单引号和双引号

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = gapfill.XGboostGapFilling(dataset=dataset, var_to_fill=var_to_fill, Mode=Mode, X_col=X_col)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run XGboost_gapfilling')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run XGboost_gapfilling: {str(e)}")

class random_forestPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'random_forest.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_random_forest_gapfilling)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_random_forest_gapfilling(self):
        # 检查是否有输入框为空
        if not self.ui.ModeEdit.text() or not self.ui.var_to_fillEdit.text() or not self.ui.X_colEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            var_to_fill = self.ui.var_to_fillEdit.text()
            Mode = self.ui.ModeEdit.text()
            X_col_str = self.ui.X_colEdit.text()
            # 将字符串转为列表
            X_col = X_col_str.split(", ")
            X_col = [item.strip("'\"") for item in X_col]# 去除每个元素的单引号和双引号

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = gapfill.RandomForestGapFilling(dataset=dataset, var_to_fill=var_to_fill, Mode=Mode, X_col=X_col)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run random_forest_gapfilling')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run random_forest_gapfilling: {str(e)}")

class AdaboostPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'Adaboost.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_Adaboost_gapfilling)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_Adaboost_gapfilling(self):
        # 检查是否有输入框为空
        if not self.ui.ModeEdit.text() or not self.ui.var_to_fillEdit.text() or not self.ui.X_colEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            var_to_fill = self.ui.var_to_fillEdit.text()
            Mode = self.ui.ModeEdit.text()
            X_col_str = self.ui.X_colEdit.text()
            # 将字符串转为列表
            X_col = X_col_str.split(", ")
            X_col = [item.strip("'\"") for item in X_col]# 去除每个元素的单引号和双引号

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = gapfill.AdaboostGapFilling(dataset=dataset, var_to_fill=var_to_fill, Mode=Mode, X_col=X_col)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run Adaboost_gapfilling')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run Adaboost_gapfilling: {str(e)}")

class ANNPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'ANN.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_ANN_gapfilling)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_ANN_gapfilling(self):
        # 检查是否有输入框为空
        if not self.ui.ModeEdit.text() or not self.ui.var_to_fillEdit.text() or not self.ui.X_colEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            var_to_fill = self.ui.var_to_fillEdit.text()
            Mode = self.ui.ModeEdit.text()
            X_col_str = self.ui.X_colEdit.text()
            # 将字符串转为列表
            X_col = X_col_str.split(", ")
            X_col = [item.strip("'\"") for item in X_col]# 去除每个元素的单引号和双引号

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")

            try:
                dataset = pd.read_csv(indir, header=0)
                data = gapfill.ANNGapFilling(dataset=dataset, var_to_fill=var_to_fill, Mode=Mode, X_col=X_col)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                        data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run ANN_gapfilling')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run ANN_gapfilling: {str(e)}")

class GapFillColsPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'GapFillCols.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_GapFillCols)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_GapFillCols(self):
        # 检查是否有输入框为空
        if not self.ui.gap_filling_methodEdit.text().strip() or not self.ui.num_to_fill_naEdit.text().strip() or not self.ui.ModeEdit.text() or not self.ui.vars_to_fillEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            outdir = self.ui.outfileEdit.text()
            gap_filling_method = self.ui.gap_filling_methodEdit.text()
            num_to_fill_na = int(self.ui.num_to_fill_naEdit.text())
            Mode = self.ui.ModeEdit.text()
            vars_to_fill_str = self.ui.vars_to_fillEdit.text()
            X_col_str = self.ui.X_colEdit.text()
            # 将字符串转为列表
            vars_to_fill = vars_to_fill_str.split(", ")
            vars_to_fill = [item.strip("'\"") for item in vars_to_fill]# 去除每个元素的单引号和双引号
            X_col = X_col_str.split(", ")
            X_col = [item.strip("'\"") for item in X_col]# 去除每个元素的单引号和双引号
            if X_col_str == '':
                X_col = []
            if vars_to_fill_str == '':
                vars_to_fill = []

            if not outdir.strip():
                QMessageBox.warning(self, "Warning", "Output file path is empty. The results will not be saved to a file.")
            #运行
            try:
                dataset = pd.read_csv(indir, header=0)
                data = gapfill.GapFillCols(dataset, gap_filling_method=gap_filling_method, num_to_fill_na=num_to_fill_na, Mode=Mode, vars_to_fill=vars_to_fill, X_col=X_col)
                if outdir:  # 只有当输出文件路径非空时才尝试保存
                    data.to_csv(outdir, index=False, mode="w")
                QMessageBox.about(self.ui, 'Message','successfully run Muti-Columns GapFill')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run Muti-Columns GapFill: {str(e)}")

##-------------------- U* Threshold Estimate --------------------
class MPTPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'MPT.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        #打开文件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))

        #添加删除日期
        self.ui.addButton.clicked.connect(self.add_time_range)
        self.ui.deleteButton.clicked.connect(self.delete_selected_range)

        #运行程序
        self.ui.runButton.clicked.connect(self.run_MPT)

    def add_time_range(self):
        """添加时间段到列表"""
        start_time = self.ui.startTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end_time = self.ui.endTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss")

        # 检查时间段是否有效
        if self.ui.startTimeEdit.dateTime() >= self.ui.endTimeEdit.dateTime():
            QMessageBox.warning(self, "Invalid Range", "Start time must be earlier than end time!")
            return

        # 添加到列表
        time_range = f"({start_time}, {end_time})"
        self.ui.timeRangeList.addItem(QListWidgetItem(time_range))

    def delete_selected_range(self):
        """删除选中的时间段"""
        selected_items = self.ui.timeRangeList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a time range to delete!")
            return
        for item in selected_items:
            self.ui.timeRangeList.takeItem(self.ui.timeRangeList.row(item))

    def get_time_ranges(self):
        """返回时间段列表"""
        ranges = []
        for index in range(self.ui.timeRangeList.count()):
            item_text = self.ui.timeRangeList.item(index).text()
            start, end = item_text.strip("()").split(", ")
            ranges.append((start, end))
        return ranges

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_MPT(self):
        datadir = self.ui.infileEdit.text()
        dataset = pd.read_csv(datadir, header=0)
        # colume names
        ustar_col = self.ui.ustar_colEdit.text()
        nee_col = self.ui.nee_colEdit.text()
        tair_col = self.ui.tair_colEdit.text()
        rg_col = self.ui.rg_colEdit.text()
        datetime_col = self.ui.datetime_colEdit.text()
        # prams
        temp_groups = int(self.ui.temp_groupsEdit.text())
        nee_bins = int(self.ui.nee_binsEdit.text())
        n_bootstraps = int(self.ui.n_bootstrapsEdit.text())
        vegetation_type = self.ui.vegetation_typeEdit.text()
        random_seed = int(self.ui.random_seedEdit.text())
        # season
        season = self.get_time_ranges()

        try:
            # Call the Bootstrap function
            _, u5, u95, u50, seasonal_results = ute.Bootstrap_ustar_threshold(
                dataset, 
                ustar_col=ustar_col, nee_col=nee_col, tair_col=tair_col, rg_col=rg_col, datetime_col=datetime_col, 
                season=season, temp_groups=temp_groups, nee_bins=nee_bins, n_bootstraps=n_bootstraps, vegetation_type=vegetation_type, random_seed=random_seed
            )

            # Format the results into a readable string
            results_msg = f"Total U* Thresholds:\n - u5 (5%): {u5:.3f}\n - u50 (median): {u50:.3f}\n - u95 (95%): {u95:.3f}\n\n"
            
            if seasonal_results:
                results_msg += "Seasonal Results:\n"
                for season_range, result in seasonal_results.items():
                    results_msg += f"  Season:\n{season_range}\n  U* threshold:\nu5={result['u5']:.3f}, u50={result['u50']:.3f}, u95={result['u95']:.3f}\n"

            # Display the results in a QMessageBox
            QMessageBox.about(self.ui, 'MPT Output', f"Successfully run MPT U* threshold estimation:\n\n{results_msg}")
        except Exception as e:
            QMessageBox.critical(self.ui, 'Error', f"Failed to run MPT U* threshold estimation: {str(e)}")

##-------------------- visualization --------------------
class plot_fingerprintPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'plot_fingerprint.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
    
        self.ui.startButton.clicked.connect(self.run_plot_fingerprint)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_plot_fingerprint(self):
        # 检查是否有输入框为空
        if not self.ui.datetime_colEdit.text() or not self.ui.data_colsEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            datetime_col = self.ui.datetime_colEdit.text()
            data_cols_str = self.ui.data_colsEdit.text()
            # 将字符串转为列表
            data_cols = data_cols_str.split(", ")
            data_cols = [item.strip("'\"") for item in data_cols]# 去除每个元素的单引号和双引号

            try:
                dataset = pd.read_csv(indir, header=0)
                visualization.plot_fingerprint(dataset, datetime_col, data_cols, cmap='viridis')
                QMessageBox.about(self.ui, 'Message','successfully run plot_fingerprint')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run plot_fingerprint: {str(e)}")

class plot_scatterPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'plot_scatter.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
    
        self.ui.startButton.clicked.connect(self.run_plot_scatter)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_plot_scatter(self):
        # 检查是否有输入框为空
        if not self.ui.datetime_colEdit.text() or not self.ui.data_colsEdit.text():
            QMessageBox.warning(self, "Warning", "Please fill in all parameters.")
        else:
            indir = self.ui.infileEdit.text()
            datetime_col = self.ui.datetime_colEdit.text()
            data_cols_str = self.ui.data_colsEdit.text()
            # 将字符串转为列表
            data_cols = data_cols_str.split(", ")
            data_cols = [item.strip("'\"") for item in data_cols]# 去除每个元素的单引号和双引号

            try:
                dataset = pd.read_csv(indir, header=0)
                visualization.plot_scatter(dataset, datetime_col, data_cols)
                QMessageBox.about(self.ui, 'Message','successfully run plot_scatter')
            except Exception as e:
                QMessageBox.critical(self.ui, 'Error', f"Failed to run plot_scatter: {str(e)}")

##-------------------- partitioning --------------------
class NT_ReichsteinPage(QWidget):
    def __init__(self):
        super().__init__()

        # 使用 QUiLoader 加载 .ui 文件并返回对应的 QWidget
        if hasattr(sys, '_MEIPASS'):  # 检测是否运行在 PyInstaller 打包环境
            base_path = sys._MEIPASS
        else:  # 开发环境
            base_path = os.path.dirname(os.path.realpath(__file__))

        # 构建 UI 文件的完整路径
        ui_path = os.path.join(base_path, 'ui', 'NT_Reichstein.ui')
        # 加载UI文件
        loader = QUiLoader()
        file = QFile(ui_path)
        if not file.open(QFile.ReadOnly):
            return None
        self.ui = loader.load(file, self)
        file.close()
        self.setLayout(self.ui.layout())  # 直接设置布局

        # 连接按钮点击事件到函数, 并传递不同的 QLineEdit 控件
        self.ui.infileButton.clicked.connect(lambda: self.open_file_dialog(self.ui.infileEdit))
        self.ui.outfileButton.clicked.connect(lambda: self.save_file_dialog(self.ui.outfileEdit))
    
        self.ui.startButton.clicked.connect(self.run_NT_Reichstein)

    def open_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def save_file_dialog(self, target_line_edit):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*);;csv Files (*.csv)")
        if file_path:
            # 将选择的文件路径设置到对应的 QLineEdit
            target_line_edit.setText(file_path)
    
    def run_NT_Reichstein(self):
        datadir = self.ui.infileEdit.text()
        data = pd.read_csv(datadir, header=0)
        # colume names
        datetime_col = self.ui.datetime_colEdit.text()
        NEE_col = self.ui.NEE_colEdit.text()
        T_col = self.ui.T_colEdit.text()
        Rg_col = self.ui.Rg_colEdit.text()
        # bool pram
        night_estimate = self.ui.night_estimatecheckBox.isChecked()
        # int prams
        min_temp_range = int(self.ui.min_temp_rangeEdit.text())
        E0_window_days = int(self.ui.E0_window_daysEdit.text())
        E0_shift_days = int(self.ui.E0_shift_daysEdit.text())
        Rref_window_days = self.ui.Rref_window_daysEdit.text()
        Rref_shift_days = int(self.ui.Rref_shift_daysEdit.text())

        try:
            partitioning.NT_Reichstein(data, datetime_col, NEE_col, T_col, Rg_col, night_estimate, min_temp_range, 
                                       E0_window_days, E0_shift_days, Rref_window_days, Rref_shift_days)
            QMessageBox.about(self.ui, 'Message','successfully run plot_scatter')
        except Exception as e:
            QMessageBox.critical(self.ui, 'Error', f"Failed to run plot_scatter: {str(e)}")


# 设置 Qt::AA_ShareOpenGLContexts 属性
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
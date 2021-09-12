# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import main
import torch.nn.functional as F
import interact
import UI.image_rc
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch
import transformers


history=[]
class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(841, 727)
        MainWindow.setMinimumSize(QtCore.QSize(841, 727))
        MainWindow.setMaximumSize(QtCore.QSize(841, 727))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/t0135e158d53957130a.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(580, 640, 81, 91))
        self.pushButton.setStyleSheet("QPushButton{border-image: url(:/1.jpg);}")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(0, 640, 581, 91))
        self.textEdit.setObjectName("textEdit")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 841, 51))
        self.textBrowser.setObjectName("textBrowser")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(1, 51, 659, 589))
        self.scrollArea.setMinimumSize(QtCore.QSize(659, 589))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 636, 588))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents)
        self.textBrowser_2.setGeometry(QtCore.QRect(0, 0, 661, 591))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(660, 30, 181, 271))
        self.label.setStyleSheet("image: url(:/2.jpg);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.scrollArea.raise_()
        self.textBrowser.raise_()
        self.textEdit.raise_()
        self.label.raise_()
        self.pushButton.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(lambda:self.show(history))
        

    def show(self,history):

        text = self.textEdit.toPlainText()
        self.textEdit.clear()
        if text:
            self.textBrowser_2.append('<font color=\"#00FF00\">User: </font>'+'<font color=\"#00FF00\">'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'</font>'+'\n')
            self.textBrowser_2.append(text)

            args = interact.set_interact_args()
            # logger = interact.create_logger(args)
            # when user choose GPU to run this programme, as well as GPU is available
            args.cuda = torch.cuda.is_available() and not args.no_cuda
            device = 'cuda' if args.cuda else 'cpu'
            tokenizer = BertTokenizer(vocab_file=args.voca_path)
            model = transformers.models.gpt2.GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
            model.to(device)
            model.eval()

            history.append(tokenizer.encode(text))
            input_ids = [tokenizer.cls_token_id]  # each input's frist token is[CLS]

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            curr_input_tensor = torch.tensor(input_ids).long().to(device)
            generated = []
            # mostly generate max_len of token
            for _ in range(args.max_len):
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = interact.top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

            history.append(generated)
            text = tokenizer.convert_ids_to_tokens(generated)
            self.textBrowser_2.append('<font color=\"#0000FF\">ChatBot: </font>'+'<font color=\"#0000FF\">'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'</font>'+'\n')
            self.textBrowser_2.append("".join(text))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "聊天机器人"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\"> </span><span style=\" font-size:14pt; font-weight:600;\">ChatBot</span></p></body></html>"))


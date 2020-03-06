import unittest
import app
import numpy as np
import tensorflow as tf
import os
from model import Deeplabv3


class TestStringMethods(unittest.TestCase):

    # --------------------------------------
    #             Test Losses
    # --------------------------------------
    def test_losses_iou_loss(self):
        # GIVEN
        y_true = np.zeros((255, 255))
        y_true[100:200, 150:200] = 1  # 5000 points area

        y_pred1 = np.zeros((255, 255))
        y_pred1[100:150, 150:200] = 1  # 2500 points area - IOU = 1 - (2500 + 10)/(2500 + 5000 - 2500 + 10) smooth=10

        y_pred2 = np.zeros((255, 255))
        y_pred2[50:60, 150:200] = 1  # 500 points area - IOU = 1 - (10)/(500 + 5000 + 10) smooth=10

        y_pred3 = np.zeros((255, 255))  # 0 point area - IOU = 1 - (10)/(5000 + 10) smooth=10

        expected_res_case_one = (1 - (2500 + 10) / (2500 + 5000 - 2500 + 10))
        expected_res_case_two = (1 - (10) / (500 + 5000 + 10))
        expected_res_case_three = (1 - (10) / (5000 + 10))

        # WHEN
        res_1 = tf.keras.backend.get_value(app.iou_loss(y_true, y_pred1, smooth=10))
        res_2 = tf.keras.backend.get_value(app.iou_loss(y_true, y_pred2, smooth=10))
        res_3 = tf.keras.backend.get_value(app.iou_loss(y_true, y_pred3, smooth=10))

        # THEN
        self.assertEquals(res_1, expected_res_case_one)
        self.assertEquals(res_2, expected_res_case_two)
        self.assertEquals(res_3, expected_res_case_three)

    def test_losses_iou_metric(self):
        # GIVEN
        y_true = np.zeros((255, 255))
        y_true[100:200, 150:200] = 1  # 5000 points area

        y_pred1 = np.zeros((255, 255))
        y_pred1[100:150, 150:200] = 1  # 2500 points area - IOU = 1 - (2500 + 10)/(2500 + 5000 - 2500 + 10) smooth=10

        y_pred2 = np.zeros((255, 255))
        y_pred2[50:60, 150:200] = 1  # 500 points area - IOU = 1 - (10)/(500 + 5000 + 10) smooth=10

        y_pred3 = np.zeros((255, 255))  # 0 point area - IOU = 1 - (10)/(5000 + 10) smooth=10

        expected_res_case_one = ((2500 + 1) / (2500 + 5000 - 2500 + 1))
        expected_res_case_two = ((1) / (500 + 5000 + 1))
        expected_res_case_three = ((1) / (5000 + 1))

        # WHEN
        res_1 = tf.keras.backend.get_value(app.iou_metric(y_true, y_pred1, smooth=1))
        res_2 = tf.keras.backend.get_value(app.iou_metric(y_true, y_pred2, smooth=1))
        res_3 = tf.keras.backend.get_value(app.iou_metric(y_true, y_pred3, smooth=1))

        # THEN
        self.assertEquals(res_1, expected_res_case_one)
        self.assertEquals(res_2, expected_res_case_two)
        self.assertEquals(res_3, expected_res_case_three)

    def test_losses_sym_dyff(self):
        # GIVEN
        y_true = np.zeros((255, 255))
        y_true[100:200, 150:200] = 1  # 5000 points area

        y_pred1 = np.zeros((255, 255))
        y_pred1[100:150, 150:200] = 1  # 2500 points area - IOU = 1 - (2500 + 10)/(2500 + 5000 - 2500 + 10) smooth=10

        y_pred2 = np.zeros((255, 255))
        y_pred2[50:60, 150:200] = 1  # 500 points area - IOU = 1 - (10)/(500 + 5000 + 10) smooth=10

        y_pred3 = np.zeros((255, 255))  # 0 point area - IOU = 1 - (10)/(5000 + 10) smooth=10

        expected_res_case_one = ((2500 + 5000 - 2*2500))
        expected_res_case_two = ((500 + 5000))
        expected_res_case_three = ((5000))

        # WHEN
        res_1 = tf.keras.backend.get_value(app.sym_dif(y_true, y_pred1))
        res_2 = tf.keras.backend.get_value(app.sym_dif(y_true, y_pred2))
        res_3 = tf.keras.backend.get_value(app.sym_dif(y_true, y_pred3))

        # THEN
        self.assertEquals(res_1, expected_res_case_one)
        self.assertEquals(res_2, expected_res_case_two)
        self.assertEquals(res_3, expected_res_case_three)

    def test_model_is_here(self):
        self.assertTrue(os.path.exists("deeplabv3_v4.h5"))

    def test_model(self):
        model = Deeplabv3(input_shape=(256, 256, 3),
                          classes=1,
                          weights='cityscapes',
                          activation='sigmoid',
                          backbone='xception')

        model.compile(optimizer=tf.optimizers.Adam(1e-4), loss=app.iou_loss, metrics=[app.iou_metric, app.sym_dif])

        #model.output.shape == tf.TensorShape([None, 256, 256, 1])

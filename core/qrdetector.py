import math
from enum import Enum

import cv2
import numpy as np
from skimage.draw import line


class ResizeDirection(Enum):
    ZOOMING = 1
    SHRINKING = 2
    UNCHANGED = 3


class QRDetector:
    def __init__(self):
        self.eps_vertical = 0.2
        self.eps_horizontal = 0.1
        self.purpose = None
        self.coeff_expansion = 0
        self.ZOOMING = ResizeDirection.ZOOMING
        self.SHRINKING = ResizeDirection.SHRINKING
        self.UNCHANGED = ResizeDirection.UNCHANGED
        self.localization_points = None

    def localization(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = self.initialize(gray)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 83, 2)
        list_lines_x = self.search_horizontal_lines(binary)
        if len(list_lines_x) == 0:
            return None
        list_lines_y = self.separate_vertical_lines(binary, list_lines_x)
        if len(list_lines_y) == 0:
            return None

        compactness, labels, self.localization_points = cv2.kmeans(list_lines_y, 3, None,
                                                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                                    10,
                                                                    0.1),
                                                                   3, cv2.KMEANS_PP_CENTERS)

        self.localization_points = self.fixationPoints(binary, self.localization_points)

        square_flag = False
        local_points_flag = False
        triangle_sides = list()
        if len(self.localization_points) == 3:
            triangle_sides.append(np.linalg.norm(self.localization_points[0] - self.localization_points[1]))
            triangle_sides.append(np.linalg.norm(self.localization_points[1] - self.localization_points[2]))
            triangle_sides.append(np.linalg.norm(self.localization_points[2] - self.localization_points[0]))

            triangle_perim = (triangle_sides[0] + triangle_sides[1] + triangle_sides[2]) / 2

            square_area = math.sqrt((triangle_perim * (triangle_perim - triangle_sides[0])
                                     * (triangle_perim - triangle_sides[1])
                                     * (triangle_perim - triangle_sides[2]))) * 2
            img_square_area = binary.shape[0] * binary.shape[1]

            if square_area > (img_square_area * 0.2):
                square_flag = True
        else:
            local_points_flag = True
        if (square_flag or local_points_flag) and self.purpose == self.SHRINKING:
            binary = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 83, 2)
            list_lines_x = self.search_horizontal_lines(binary)
            if len(list_lines_x) == 0:
                return None
            list_lines_y = self.separate_vertical_lines(binary, list_lines_x)
            if len(list_lines_y) == 0:
                return None

            compactness, labels, self.localization_points = cv2.kmeans(list_lines_y, 3, None,
                                                                       (
                                                                           cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                                           10,
                                                                           0.1),
                                                                       3, cv2.KMEANS_PP_CENTERS)
            self.localization_points = self.fixationPoints(binary, self.localization_points)
            if len(self.localization_points) != 3:
                return None

            new_size = np.round(
                [binary.shape[0] * self.coeff_expansion, binary.shape[1] * self.coeff_expansion]).astype(np.uint)
            binary = cv2.resize(binary, new_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            for i in range(len(self.localization_points)):
                self.localization_points[i] *= self.coeff_expansion
        if self.purpose == self.ZOOMING:
            new_size = np.round(
                [binary.shape[0] / self.coeff_expansion, binary.shape[1] / self.coeff_expansion]).astype(np.uint)
            binary = cv2.resize(binary, new_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            for i in range(len(self.localization_points)):
                self.localization_points[i] /= self.coeff_expandsion

            for i in range(len(self.localization_points)):
                for j in range(i + 1, len(self.localization_points)):
                    if np.linalg.norm(self.localization_points[i] - self.localization_points[j]) < 10:
                        return None

        self.localization_points = np.asarray(self.localization_points)
        return self.localization_points

    def compute_transformation_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 83, 2)
        if len(self.localization_points) != 3:
            return None

        non_zero_elem = list()
        new_non_zero_elem = list()
        for i in range(3):
            mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2)).astype(np.uint8)
            future_pixel = 255
            count_test_lines = 0
            for index in range(np.round(self.localization_points[i][0]).astype(np.int), image.shape[0] - 1):
                next_pixel = binary[index + 1, np.round(self.localization_points[i][1].astype(np.int))]
                if next_pixel == future_pixel:
                    future_pixel = 0 if future_pixel == 255 else 255
                    count_test_lines += 1
                    if count_test_lines == 2:
                        cv2.floodFill(binary, mask,
                                      [index + 1, np.round(self.localization_points[i][1].astype(np.int))], 255, 0,
                                      cv2.FLOODFILL_MASK_ONLY)
                        break
            mask_roi = mask[1: binary.shape[0] - 1, 1: binary.shape[1] - 1]
            non_zero_elem.append(cv2.findNonZero(mask_roi))
        newHull = np.concatenate([non_zero_elem[0], non_zero_elem[1], non_zero_elem[2]], axis=0)
        locations = cv2.convexHull(newHull)

        rect = cv2.minAreaRect(locations)
        transformation_points = cv2.boxPoints(rect)

        # for location in locations:
        #     cv2.circle(image, location.squeeze(), 1, (0, 0, 255), 25)
        #
        # image = cv2.resize(image, (512, 512))
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # for i in range(len(locations)):
        #     for j in range(3):
        #         for k in range(len(non_zero_elem[j])):
        #             if locations[i][0, 0] == non_zero_elem[j][k][0, 0] and locations[i][0, 1] == \
        #                     non_zero_elem[j][k][0, 1]:
        #                 new_non_zero_elem.append(locations[i])
        #
        # pentagon_diag_norm = -1
        # for i in range(len(new_non_zero_elem[1])):
        #     for j in range(len(new_non_zero_elem[2])):
        #         temp_norm = np.linalg.norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j])
        #         if temp_norm > pentagon_diag_norm:
        #             down_left_edge_point = new_non_zero_elem[1][i]
        #             up_right_edge_point = new_non_zero_elem[2][j]
        #             pentagon_diag_norm = temp_norm
        # if (down_left_edge_point[0] == 0 and down_left_edge_point[1] == 0) or (
        #         up_right_edge_point[0] == 0 and up_right_edge_point[1] == 0) or len(new_non_zero_elem[0]) == 0:
        #     return None
        #
        # max_area = -1
        # up_left_edge_point = new_non_zero_elem[0][0]
        #
        # for i in range(len(new_non_zero_elem[0])):
        #     list_edge_points = list()
        #     list_edge_points.append(new_non_zero_elem[0][i])
        #     list_edge_points.append(down_left_edge_point)
        #     list_edge_points.append(up_right_edge_point)
        #
        #     temp_area = math.fabs(cv2.contourArea(np.asarray(list_edge_points)))
        #     if max_area < temp_area:
        #         up_left_edge_point = new_non_zero_elem[0][i]
        #         max_area = temp_area
        #
        #     norm_down_max_delta = -1
        #     norm_up_max_delta = -1
        #     for i in range(len(new_non_zero_elem[1])):
        #         temp_norm_delta = np.linalg.norm(up_left_edge_point - new_non_zero_elem[1][i]) + np.linalg.norm(
        #             down_left_edge_point - new_non_zero_elem[1][i])
        #         if norm_down_max_delta < temp_norm_delta:
        #             down_max_delta_point = new_non_zero_elem[1][i]
        #             norm_down_max_delta = temp_norm_delta
        #
        #     for i in range(len(new_non_zero_elem[2])):
        #         temp_norm_delta = np.linalg.norm(up_left_edge_point - new_non_zero_elem[2][i]) + np.linalg.norm(
        #             up_right_edge_point - new_non_zero_elem[2][i])
        #         if norm_up_max_delta < temp_norm_delta:
        #             up_max_delta_point = new_non_zero_elem[2][i]
        #             norm_up_max_delta = temp_norm_delta
        #
        #     transformation_points = list()
        #     transformation_points.append(down_max_delta_point)
        #     transformation_points.append(up_left_edge_point)
        #     transformation_points.append(up_right_edge_point)
        #     transformation_points.append(
        #         self.intersection_lines(down_left_edge_point, down_max_delta_point,
        #                                 up_right_edge_point, up_max_delta_point)
        #     )
        #     # transformation_points = self.get_quadrilateral(binary, transformation_points)
        #
        #     width, height = binary.shape[0], binary.shape[1]
        #     for i in range(len(transformation_points)):
        #         if np.round(transformation_points[i][0]) > width or np.round(transformation_points[i][1]) > height:
        #             return None

        return np.asarray(transformation_points)

    def initialize(self, image):
        min_side = min(image.shape[0], image.shape[1])
        if min_side < 512:
            self.purpose = self.ZOOMING
            self.coeff_expansion = 512.0 / min_side
            new_size = np.round([image.shape[0] * self.coeff_expansion, image.shape[1] * self.coeff_expansion]).astype(
                np.uint)
            image = cv2.resize(image, new_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        elif min_side > 512:
            self.purpose = self.SHRINKING
            self.coeff_expansion = min_side / 512.0
            new_size = np.round([image.shape[0] / self.coeff_expansion, image.shape[1] / self.coeff_expansion]).astype(
                np.uint)
            image = cv2.resize(image, new_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)
        else:
            self.purpose = self.UNCHANGED
            self.coeff_expansion = 1.0
        return image

    def search_horizontal_lines(self, image):
        result = list()
        height = image.shape[1]
        width = image.shape[0]

        for y in range(height):
            pixels_position = list()
            image_row = image[:, y]

            for pos in range(width):
                if image_row[pos] == 0:
                    break
            if pos == width:
                continue

            pixels_position.append(pos)
            pixels_position.append(pos)
            pixels_position.append(pos)

            future_pixel = 255
            for x in range(pos, width):
                if image_row[x] == future_pixel:
                    future_pixel = 0 if future_pixel == 255 else 255
                    pixels_position.append(x)
            pixels_position.append(width - 1)
            for i in range(2, len(pixels_position) - 4, 2):
                test_lines = list()
                test_lines.append(pixels_position[i - 1] - pixels_position[i - 2])
                test_lines.append(pixels_position[i] - pixels_position[i - 1])
                test_lines.append(pixels_position[i + 1] - pixels_position[i])
                test_lines.append(pixels_position[i + 2] - pixels_position[i + 1])
                test_lines.append(pixels_position[i + 3] - pixels_position[i + 2])

                length = np.sum(test_lines)
                if length == 0:
                    continue

                weight = 0
                for j in range(len(test_lines)):
                    if j != 2:
                        weight += math.fabs((test_lines[j] / length) - 1.0 / 7.0)
                    else:
                        weight += math.fabs((test_lines[j] / length) - 3.0 / 7.0)
                if weight < self.eps_vertical:
                    line = list()
                    line.append(pixels_position[i - 2])
                    line.append(y)
                    line.append(length)
                    result.append(line)
        return np.asarray(result).astype(np.float32)

    def separate_vertical_lines(self, image, list_lines):
        min_dist_between_points = 10.0
        max_ratio = 1.0
        for coeff_epsilon_i in range(1, 101):
            coeff_epsilon = coeff_epsilon_i * 0.1
            point2f_result = self.extract_vertical_lines(image, list_lines, self.eps_horizontal * coeff_epsilon)
            if len(point2f_result) != 0:
                compactness, labels, centers = cv2.kmeans(point2f_result, 3, None,
                                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
                                                          3, cv2.KMEANS_PP_CENTERS)
                min_dist = 1.79769e+308
                for i in range(len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[(i + 1) % len(centers)])
                    if dist < min_dist:
                        min_dist = dist
                if min_dist < min_dist_between_points:
                    continue
                mean_compactness = compactness / len(point2f_result)
                ratio = mean_compactness / min_dist

                if ratio < max_ratio:
                    return point2f_result
        return np.array([])

    def extract_vertical_lines(self, image, list_lines, eps):
        result = list()

        for pnt in range(len(list_lines)):
            x, y = np.round([list_lines[pnt][0] + list_lines[pnt][2] * 0.5, list_lines[pnt][1]]).astype(np.uint)

            # Search vertical up-lines
            test_lines = list()
            future_pixel_up = 255
            temp_length_up = 0
            for j in range(y, image.shape[1] - 1):
                next_pixel = image[x, j + 1]
                temp_length_up += 1
                if next_pixel == future_pixel_up:
                    future_pixel_up = 0 if future_pixel_up == 255 else 255
                    test_lines.append(temp_length_up)
                    temp_length_up = 0
                    if len(test_lines) == 3:
                        break

            # Search vertical down-lines
            future_pixel_down = 255
            temp_length_down = 0
            for j in range(y, 0, -1):
                next_pixel = image[x, j - 1]
                temp_length_down += 1
                if next_pixel == future_pixel_down:
                    future_pixel_down = 0 if future_pixel_down == 255 else 255
                    test_lines.append(temp_length_down)
                    temp_length_down = 0
                    if len(test_lines) == 6:
                        break

            # Compute vertical lines
            if len(test_lines) == 6:
                length = np.sum(test_lines)

                assert length > 0, "Length should be greater than 0."
                weight = 0
                for i in range(len(test_lines)):
                    if i % 3 != 0:
                        weight += math.fabs((test_lines[i] / length) - 1.0 / 7.0)
                    else:
                        weight += math.fabs((test_lines[i] / length) - 3.0 / 14.0)

                if weight < eps:
                    result.append(list_lines[pnt])

        point2f_result = list()
        if len(result) > 2:
            for i in range(len(result)):
                point2f_result.append([result[i][0] + result[i][2] * 0.5, result[i][1]])
        return np.asarray(point2f_result).astype(np.float32)

    def fixationPoints(self, image, local_point):
        norm_triangl = list()
        cos_angles = list()

        norm_triangl.append(np.linalg.norm(local_point[1] - local_point[2]))
        norm_triangl.append(np.linalg.norm(local_point[0] - local_point[2]))
        norm_triangl.append(np.linalg.norm(local_point[1] - local_point[0]))

        cos_angles.append((norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2] - norm_triangl[0] *
                           norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]))
        cos_angles.append((norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2] - norm_triangl[1] *
                           norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]))
        cos_angles.append((norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1] - norm_triangl[2] *
                           norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]))

        angle_barrier = 0.85
        if math.fabs(cos_angles[0]) > angle_barrier or math.fabs(cos_angles[1]) > angle_barrier or math.fabs(
                cos_angles[2]) > angle_barrier:
            return np.array([])

        if cos_angles[0] < cos_angles[1] and cos_angles[0] < cos_angles[2]:
            i_min_cos = 0
        elif cos_angles[1] < cos_angles[0] and cos_angles[1] < cos_angles[2]:
            i_min_cos = 1
        else:
            i_min_cos = 2

        index_max = 0
        max_area = 2.22507e-308
        for i in range(len(local_point)):
            current_index = i % 3
            left_index = (i + 1) % 3
            right_index = (i + 2) % 3

            current_point = local_point[current_index]
            left_point = local_point[left_index]
            right_point = local_point[right_index]
            central_point = self.intersection_lines(current_point,
                                                    [(local_point[left_index][0] + local_point[right_index][0]) * 0.5,
                                                     (local_point[left_index][1] + local_point[right_index][1]) * 0.5],
                                                    [0, image.shape[1] - 1], [image.shape[0] - 1, image.shape[1] - 1])
            list_area_pnt = list()
            list_area_pnt.append(current_point)

            list_line_iter = list()
            list_line_iter.append(np.asarray(
                list(zip(*line(int(current_point[0]), int(current_point[1]), int(left_point[0]), int(left_point[1]))))))
            list_line_iter.append(np.asarray(list(
                zip(*line(int(current_point[0]), int(current_point[1]), int(central_point[0]),
                          int(central_point[1]))))))
            list_line_iter.append(np.asarray(list(
                zip(*line(int(current_point[0]), int(current_point[1]), int(right_point[0]), int(right_point[1]))))))

            for k in range(len(list_line_iter)):
                li = list_line_iter[k]
                future_pixel = 255
                count_index = 0
                for j in range(len(li)):
                    p = li[j]
                    if p[0] >= image.shape[0] or p[1] >= image.shape[1]:
                        break

                    value = image[p[0], p[1]]
                    if value == future_pixel:
                        future_pixel = 0 if future_pixel == 255 else 255
                        count_index += 1
                        if count_index == 3:
                            list_area_pnt.append(p)
                            break

                temp_check_area = cv2.contourArea(np.asarray(list_area_pnt).astype(np.float32))
                if temp_check_area > max_area:
                    index_max = current_index
                    max_area = temp_check_area
        if index_max == i_min_cos:
            local_point[0], local_point[index_max] = local_point[index_max], local_point[0]
        else:
            return np.array([])

        rpt = local_point[0]
        bpt = local_point[1]
        gpt = local_point[2]
        m = [[rpt[0] - bpt[0], rpt[1] - bpt[1]],
             [gpt[0] - rpt[0], gpt[1] - rpt[1]]]
        if np.linalg.det(m) > 0:
            local_point[1], local_point[2] = local_point[2], local_point[1]

        return np.asarray(local_point)

    def intersection_lines(self, a1, a2, b1, b2):
        divisor = (a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0])
        eps = 0.001
        if abs(divisor) < eps:
            return a2
        result_square_angle = [((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0]) -
                                (b1[0] * b2[1] - b1[1] * b2[0]) * (a1[0] - a2[0])) /
                               divisor,
                               ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1]) -
                                (b1[0] * b2[1] - b1[1] * b2[0]) * (a1[1] - a2[1])) /
                               divisor]
        return result_square_angle

    def get_quadrilateral(self, image, angle_list):
        pass

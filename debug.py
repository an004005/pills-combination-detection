import cv2
import numpy as np
import copy
import os
import math

MOMENT_THRESH = 0.2
SE1 = np.ones((3, 3), np.uint8)
SE2 = np.ones((5, 5), np.uint8)
SE3 = np.ones((13, 13), np.uint8)
RESIZE_WIDTH = 600


def type_contour(path):
    type_img = cv2.imread(path)
    sstype_img = cv2.cvtColor(type_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(sstype_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(type_img, [contours[0]], 0, [255,255,255], 1)
    # cv2.imshow('d', type_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return contours[0]


pill_types = {
    'type1': type_contour("types/type1.png"),
    'type2': type_contour("types/type2.png"),
    'type3': type_contour("types/type3.png")
}


def binaryImageCanny(grayImage):
    canny = cv2.Canny(grayImage, 20, 150)

    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, SE1)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, SE2)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(canny, contours, -1, (255, 255, 255), -1)

    return canny


def binaryImageAdaptiveThresh(grayImage, c=2):
    blured = cv2.medianBlur(grayImage, 5)
    thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, c)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, SE1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, SE2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, SE1)

    thresh = cv2.bitwise_not(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(thresh, [contour], 0, (255, 255, 255), -1)
    thresh = cv2.bitwise_not(thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, SE3)

    thresh = cv2.bitwise_not(thresh)

    return thresh


def resizeFixRate(image, width):
    img_width = image.shape[1]
    imgRatio = width / img_width
    return cv2.resize(image, dsize=(0, 0), fx=imgRatio, fy=imgRatio, interpolation=cv2.INTER_LINEAR)


def getContours(binaryImage):
    return cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def show(image_to_show):
    cv2.imshow("image to show", image_to_show)
    cv2.waitKey()
    cv2.destroyAllWindows()


def compute_contour_center(contour):
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def min_max_from_contour(contour, center):
    min_d = 99999
    max_d = -1
    center_x = center[0]
    center_y = center[1]
    for point in contour:
        x_diff = center_x - point[0][0]
        y_diff = center_y - point[0][1]
        distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
        if min_d >= distance:
            min_d = distance

        if max_d <= distance:
            max_d = distance
    return min_d, max_d


def register_pills_image(image_path):
    pills_combination = {}
    pills_combination['name'] = os.path.basename(open(image_path).name)

    tmpl_image = cv2.imread(image_path)
    tmpl_image = resizeFixRate(tmpl_image, RESIZE_WIDTH)
    pills_combination['image'] = copy.deepcopy(tmpl_image)

    tmpl_gray = cv2.cvtColor(tmpl_image, cv2.COLOR_BGR2GRAY)
    tmpl_binary = binaryImageCanny(tmpl_gray)
    # show(tmpl_binary)
    tmpl_contours, _ = getContours(tmpl_binary)

    pills_combination['contours'] = []
    pills_combination['center'] = []
    pills_combination['min_max_size'] = []
    pills_combination['pill_name'] = []

    # test_image = copy.deepcopy(tmpl_image)
    # cv2.drawContours(test_image, tmpl_contours, -1, [255,255,255], 1)
    # show(test_image)

    # 잡음 제거 및 알약의 중심점, 크기 정리
    max_contour_len = len(max(tmpl_contours, key=lambda x: len(x)))
    for i, contour in enumerate(tmpl_contours):
        if len(contour) < max_contour_len / 4:  # 그림자, 잡음으로 생긴 blob제거 따라서 사진을 최대한 알약이 크게 보이게 찍는다.
            continue
        pills_combination['contours'].append(contour)
        center = compute_contour_center(contour)
        pills_combination['center'].append(center)
        pills_combination['min_max_size'].append(min_max_from_contour(contour, center))

        # 알약 확인용
        # tmp = copy.deepcopy(tmpl_image)
        # cv2.drawContours(tmp, [contour], 0, [255, 255, 255], 1)
        # cv2.imshow(pills_combination['name'] + " blob " + str(i), tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 입력받은 알약들을 type별로 정리
    for type_name, type_contour in pill_types.items():
        pills_combination[type_name] = []
        for i, contour in enumerate(pills_combination['contours']):
            moment = cv2.matchShapes(type_contour, contour, cv2.CONTOURS_MATCH_I3, 0)
            if moment <= MOMENT_THRESH:
                pills_combination[type_name].append(i)

                # 확인용
                # tmp = copy.deepcopy(tmpl_image)
                # cv2.drawContours(tmp, [contour], 0, [255, 255, 255], 1)
                # cv2.imshow(pills_combination['name'] + " blob " + str(i) + type_name, tmp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    return pills_combination


def compare_pill_detail(pill1, pill2):
    return True


# def compare_pill(tmpl_pills, user_pills):
#     result = {}
#     for i, tmpl_contour in enumerate(tmpl_pills['contours']):
#         tmpl_image = copy.deepcopy(tmpl_pills['image'])
#         cv2.drawContours(tmpl_image, [tmpl_contour], 0,  [255, 255, 255], 1)
#         result[i] = []
#         for j, user_contour in enumerate(user_pills['contours']):
#             moment = cv2.matchShapes(tmpl_contour, user_contour, cv2.CONTOURS_MATCH_I3, 0)
#             if moment > MOMENT_THRESH: # MOMENT_THRESH보다 작으면 같은 모양으로 취급
#                 continue
#             # tmpl, user contour index
#             result[i].append(j)
#             user_image = copy.deepcopy(user_pills['image'])
#             cv2.drawContours(user_image, [user_contour], 0,  [255, 255, 255], 1)
#
#             cv2.imshow("tmpl" + str(moment), tmpl_image)
#             cv2.imshow("user" + str(moment), user_image)
#
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
def get_type1_radius(contour, center):
    center_x = center[0]
    center_y = center[1]
    sum_distance = 0
    for point in contour:
        x_diff = center_x - point[0][0]
        y_diff = center_y - point[0][1]
        distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
        sum_distance += distance
    return sum_distance / len(contour)


def showIndexesPill(pills, indexes, type_name):
    image = copy.deepcopy(pills['image'])
    indexes = list(map(lambda i: pills[type_name][i], indexes))
    target = list(map(lambda i: pills['contours'][i], indexes))
    cv2.drawContours(image, target, -1, [255, 255, 255], 2)
    cv2.imshow("pill", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale_compare(pills1, pills2, type_name):  # tmpl, user pills
    # 무게중심에서 각 contour의 점의 거리의 평균으로 크기를 비교한다.
    # 개수가 1개일때는 비교 불가(상대크기를 알 수 없다)
    pills1_radius = []
    pills2_radius = []
    for i in pills1[type_name]:
        center = pills1['center'][i]
        contour = pills1['contours'][i]
        pills1_radius.append(get_type1_radius(contour, center))

    for i in pills2[type_name]:
        center = pills2['center'][i]
        contour = pills2['contours'][i]
        pills2_radius.append(get_type1_radius(contour, center))
    pills1_radius_norm = list(map(lambda x: x / max(pills1_radius), pills1_radius))
    pills2_radius_norm = list(map(lambda x: x / max(pills2_radius), pills2_radius))

    NORM_THRESH = 0.1
    size_same_group = {}  # key: tmpl, value: type의 인덱스
    for i, pill1 in enumerate(pills1_radius_norm):
        size_same_group[i] = []
        for j, pill2 in enumerate(pills2_radius_norm):
            if abs(pill1 - pill2) <= NORM_THRESH:
                # same
                size_same_group[i].append(j)

    check_set = set([])
    for k, v in size_same_group.items():
        if len(v) == 0:
            showIndexesPill(pills1, [k], type_name)
            return False, size_same_group
        else:
            check_set.update(v)
    if len(check_set) != len(pills2_radius):
        type_index = set([i for i in range(len(pills2[type_name]))])
        showIndexesPill(pills2, type_index - check_set, type_name)
        return False, size_same_group

    return True, size_same_group


def showTypePills(tmpl_pills, user_pills, type_name):
    tmpl_image = copy.deepcopy(tmpl_pills['image'])
    tmpl_target = list(map(lambda i: tmpl_pills['contours'][i], tmpl_pills[type_name]))
    cv2.drawContours(tmpl_image, tmpl_target, -1, [255, 255, 255], 2)

    user_image = copy.deepcopy(user_pills['image'])
    user_target = list(map(lambda i: user_pills['contours'][i], user_pills[type_name]))
    cv2.drawContours(user_image, user_target, -1, [255, 255, 255], 2)

    cv2.imshow("tmpl_pills", tmpl_image)
    cv2.imshow("user_pills", user_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_pills(tmpl_pills, user_pills):
    # 타입별 개수 확인
    check_types = []
    for type_name in pill_types.keys():
        if len(tmpl_pills[type_name]) != len(user_pills[type_name]):
            print(type_name + " of pills count is not same")
            showTypePills(tmpl_pills, user_pills, type_name)
            return False
        elif len(tmpl_pills[type_name]) > 0 and len(tmpl_pills[type_name]) == len(user_pills[type_name]):
            check_types.append(type_name)

    # 각 타입별 크기 확인, check_types에는 type에 해당하는 알약이 1개 이상 들어있고 개수가 같음
    type_group = {}
    for type_name in check_types:
        succeeded, size_same_group = scale_compare(tmpl_pills, user_pills, type_name)
        if not succeeded:
            print(size_same_group)
            print(type_name + " of pills size is not same")
            return False
        else:
            type_group[type_name] = size_same_group

    # 각 타입별 디테일 확인
    print(type_group)

    return True


user_pills1 = register_pills_image("images/user_pills1.jpg")
user_pills6 = register_pills_image("images/user_pills6.jpg")
compare_pills(user_pills6, user_pills1)

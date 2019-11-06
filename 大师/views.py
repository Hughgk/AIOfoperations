from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.templatetags.static import static
from matplotlib import pyplot

from .utils.cnn_model import *
from .utils.image_processing import *
from .utils.calculator import *
from PIL import Image

meta = static("/model/model-200.meta")
path = static("/model/")

#TODO : 界面美化 ， 历史记录

def main_page(request):
    return render(request, "hand_writing_calculator/index.html")

def save_img(img_arr: np.ndarray, file_path: str) -> None:
    img = Image.fromarray(img_arr, 'L')
    img.save(file_path)

@csrf_exempt
def get_result(request):
    img_str = request.POST["img_data"]
    global cnn_model
    # print("img_str=",img_str)
    # print("path=",path)
    #rootname 为项目根目录 handwriting_calculator
    rootname = os.path.dirname(__file__)
    # print(os.path.dirname(__file__))
    img_arr = np.array(img_str.split(',')).reshape(200, 1000, 4).astype(np.uint8)

    binary_img_arr = img_arr[:, :, 2]

    # print ("img_arr=",img_arr)
    # print("binary_img_arr=", binary_img_arr)

    if 255 in binary_img_arr and 0 in binary_img_arr:
        save_img(binary_img_arr, "./target.png")
        data = cv2.imread('./target.png', 2)
        data = 255 - data
        # print("data=",data)
        # print("0 in data=", 0 in data)
        # print("255 in data=", 255 in data)
        # print("0 in binary_img_arr=", 0 in binary_img_arr)
        # print("255 in binary_img_arr=", 255 in binary_img_arr)
        # print("0 in img_arr=", 0 in img_arr)
        # print("255 in img_arr=", 255 in img_arr)

        images = get_image_cuts(data, is_data=True, n_lines=1, data_needed=True)
        equation = ''
        cnn_model = model()
        # cnn_model.load_model(meta, path)
        cnn_model.load_model(rootname + '/static/model/model-200.meta',rootname + '/static/model/')
        #将剪切的图片传入模型进行识别 ， 结果保存在列表中
        # print("images=" , images)
        digits = list(cnn_model.predict(images))
        for d in digits:
            equation += SYMBOL[d]
            # print("SYMBOL[d]=",SYMBOL[d])
        print(equation)
        try :
            print(eval(equation))
        except SyntaxError:
            return JsonResponse({"status": "{} ： 算式有误，请重新输入一次！".format(equation)}, safe=False)
        st = calculate(equation)
        if st != '?' :
            result = compvalue(st).pop()

        else :
            result = '未知结果'
        return JsonResponse({"status": "{} = {}".format(equation, result)}, safe=False)
    else :
        #服务器收到的是空白的图片
        return JsonResponse({"status": "未获取到图片，请重新输入"}, safe=False)


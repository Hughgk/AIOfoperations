from django.contrib.messages.storage import session
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.templatetags.static import static
from django.shortcuts import reverse,redirect
from matplotlib import pyplot

from .utils.cnn_model import *
from .utils.image_processing import *
from .utils.calculator import *
from handwriting_calculator.user import User
from PIL import Image

meta = static("/model/model-200.meta")
path = static("/model/")

i = 0

#主界面是登录界面
def main_page(request):
    return render(request, "hand_writing_calculator/index.html")

def index(request) :
    return render(request,"hand_writing_calculator/loginpage.html")

def save_img(img_arr: np.ndarray, file_path: str) -> None:
    global i
    img = Image.fromarray(img_arr, 'L')
    i +=1
    img.save(file_path)

#登录逻辑
@csrf_exempt
def login(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    form = {
        "username": username,
        "password": password,

    }
    u = User.validate_login(form)
    if u is None :
        return redirect(reverse("main_page"))
    else:
        # session 中写入 user_id
        print("u.id=",u.id)

        request.session['uid'] = u.id
        # 设置 cookie 有效期为 永久
        #request.session.permanent = True
        return redirect(reverse('main_page'))



#注册逻辑
@csrf_exempt
def register(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    form = {
        "username":username,
        "password":password,

             }
    # 用类函数来判断
    u = User.register(form)
    return redirect(reverse('index'))

#四则运算图片处理逻辑
@csrf_exempt
def get_result(request):
    img_str = request.POST["img_data"]
    global cnn_model

    rootname = os.path.dirname(__file__)
    img_arr = np.array(img_str.split(',')).reshape(200, 1000, 4).astype(np.uint8)

    binary_img_arr = img_arr[:, :, 2]

    if 255 in binary_img_arr and 0 in binary_img_arr:
        path = "./imgs/target{}.png".format(i)
        save_img(binary_img_arr, path )
        data = cv2.imread(path, 2)
        data = 255 - data


        images = get_image_cuts(data, is_data=True, n_lines=1, data_needed=True)
        print("images : ",images)
        equation = ''
        cnn_model = model()
        # cnn_model.load_model(meta, path)
        # print("rootname=",rootname + '/static/model/model-200.meta')
        cnn_model.load_model(rootname + '/static/model/model-200.meta',rootname + '/static/model/')
        #将剪切的图片传入模型进行识别 ， 结果保存在列表中
        # print("images=" , images)
        digits = list(cnn_model.predict(images))
        print("digits :" ,digits)
        for d in digits:
            equation += SYMBOL[d]
            # print("SYMBOL[d]=",SYMBOL[d])
        print(equation)
        try :
            return JsonResponse({"status": "{} = {}".format(equation,eval(equation))}, safe=False)
        except SyntaxError:
            return JsonResponse({"status": "{} ： 算式有误，请重新输入一次！".format(equation)}, safe=False)
        except ZeroDivisionError:
            return JsonResponse({"status": "{} ： 除0错误".format(equation)}, safe=False)
        except TypeError:
            return JsonResponse({"status": "{} ： 算式有误，请重新输入一次！".format(equation)}, safe=False)
        st = calculate(equation)
        if st != '?' :
            result = compvalue(st).pop()

        else :
            result = '未知结果'
        return JsonResponse({"status": "{} = {}".format(equation, result)}, safe=False)
    else :
        #服务器收到的是空白的图片
        return JsonResponse({"status": "未获取到图片，请重新输入"}, safe=False,json_dumps_params={'ensure_ascii':False})


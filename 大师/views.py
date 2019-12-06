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

def current_user(request):
    # 从 session 中找到 user_id 字段, 找不到就 -1
    # 然后 User.find_by 来用 id 找用户
    # 找不到就返回 None
    uid = request.session.get('uid', -1)
    print("current_user uid=",uid)
    u = User.find_by(id=uid)
    print("current user=",u)
    return u


#四则运算程序界面
def main_page(request):
    print("main_page=",request.session.get('uid', -1))
    user = current_user(request)
    if user == None :
        #游客
        return redirect(reverse("visitor"))
    return render(request, "hand_writing_calculator/mainpage.html",{'user':user,'uid':user.id})

#登录界面
def index(request) :
    return render(request,"hand_writing_calculator/loginpage.html")

def visitor(request):
    request.session['uid'] = -1
    return render(request, "hand_writing_calculator/mainpage.html",{'uid':-1})

def profile(request):
    #没有登录不能访问
    if request.session.get('uid', -1) == -1 :
        return redirect(reverse("index"))
    uid = request.GET.get("id","")
    print("profile uid=",uid)
    request.session['uid'] = int(uid)
    #id必须是int类型的才能找到用户
    user = current_user(request)
    print("user=",user.username)
    if user == None :
        #游客
        return redirect(reverse("visitor"))
    return render(request, "hand_writing_calculator/profile.html", {'user': user, 'uid': user.id})


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
    print("form=",form)
    u = User.validate_login(form)
    if u is None :
        return redirect(reverse("index"))
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


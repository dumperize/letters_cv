import requests
import base64
import json
import io
import random
import numpy as np
from PIL import Image
from model import get_model
from keras.models import load_model
import cv2

def preprocessing_fun(img):
#     print(img.shape, img.astype(int))
#     print(filename)
#     img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # gray = img.astype(np.uint8)
#     img = cv2.fastNlMeansDenoising(gray,None,5,7,21)
#     img = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)
    ret,thresh = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 1)
    img = cv2.bitwise_or(img, erode)
    
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(thresh, kernel, iterations = 1)
    
    dec = 60
    img = np.where(img < 250, img - dec, img)
    
    gray = 255*(img < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    
#     img = cv2.copyMakeBorder(rect,top,bottom,right,left,cv2.BORDER_CONSTANT,value=255)
    img = cv2.copyMakeBorder(rect,5,5,5,5,cv2.BORDER_CONSTANT,value=255)
    img = cv2.resize(img, (80,80), interpolation = cv2.INTER_NEAREST)
#     img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
#     gray = img.astype(np.uint8)
#     img = cv2.fastNlMeansDenoising(gray,None,5,7,21)
#     img = cv2.blur(img, (5, 5))
    img = img.reshape(80,80,1)
#     img = crop_white_space(img)
    
    return img
# ФУНКЦИИ ПОЛУЧЕНИЯ И ОТПРАВКИ ДАННЫХ =========================================================================

# получить от сервера данные вида  { "response" : { "data" : { hash : img, ... } }, "error" : error, "code" : code }
def __getData(url, user_token):
    resp = requests.post(url, json={'token': user_token})
    jsn = None
    if resp.status_code == 200: jsn = resp.json()
    return jsn


# отправить на сервер данные вида  { "token": token, "data": { hash : label, ... } }
def __sendAnswer(url, user_token, data):
    data = json.dumps(data, ensure_ascii=False)
    resp = requests.post(url, json={"token": user_token, "data": data})
    jsn = None
    if resp.status_code == 200: jsn = resp.json()
    return jsn

# ============================ КЛАССИФИКАТОР ================================
# предсказание буквы по изображению - сейчас стоит случайный генератор меток
# !!! ЗДЕСЬ ДОЛЖНО БЫТЬ ОБРАЩЕНИЕ К ВАШЕМУ КЛАССИФИКАТОРУ!!!
# !!! ВЫ МОЖЕТЕ ПЕРЕДАВАТЬ НЕ ОТДЕЛЬНЫЕ ИЗОБРАЖЕНИЯ (как здесь), а СРАЗУ ВСЮ ПОРЦИЮ
# !!! ГЛАВНОЕ, ЧТОБЫ ВЕРНУЛИСЬ МЕТКИ ДЛЯ КАЖДОГО ИЗОБРАЖЕНИЯ
def predict(img):
    model = model = load_model('weight/20-epoch')
    print(model.summary())
    img = preprocessing_fun(np.array(img))
    print(img.shape)
    prediction = model.predict(img)
    print(prediction)
    labels = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ы', 'ъ', 'э', 'ю', 'я']
    ind = np.argmax(prediction)
    return labels[ind]


# СЕАНС ТЕСТИРОВАНИЯ =========================================================================

get_url = "https://mooped.net/local/hackathon/api/data/get"
send_url = "https://mooped.net/local/hackathon/api/data/send"
# user_token = "<user_token>" # токен пользователя
user_token = "022886bb4d6ee778e07d466b3e42d7c3" # здесь должен быть ВАШ токен пользователя

while True:
    jsn = __getData(get_url, user_token) # запрашиываем данные у сервера и читаем из JSON
    if jsn != None:
        data = jsn["response"]["data"]  # извлекаем данные из словаря
        # классифицируем изображения - формируем словарь меток predicted_labels
        predicted_labels = {}
        
        for hash in data:
            img = base64.b64decode(data[hash])
            img = Image.open(io.BytesIO(img))
            # img = np.asarray(img) # тензор размерностью (128, 128, 3)
            predicted_label = predict(img)
            predicted_labels[hash] = predicted_label

        # преобразуем словарь с метками в JSON и отправляем ответ серверу
        # jsn = __sendAnswer(send_url, user_token, predicted_labels)

        if jsn == None: # проверка успешности
            print("ERROR")
            break
        else: # данные успешно доставлены - проверяем ответ от сервера
            status_code = jsn["code"]
            message = jsn["response"]["message"]
            print(message)
            if status_code == 0: break  # если все данные классифицированы, то завершить
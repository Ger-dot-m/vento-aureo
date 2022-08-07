import pandas as pd
from bs4 import BeautifulSoup
import requests
import time

url = 'https://cincodias.elpais.com/tag/tesla_motors/a/'
base_url = 'https://cincodias.elpais.com'
traceback = 'Enlaces:\n'
links = []
for i in range(31, 0, -1):
    n_page = requests.get(url+str(i))

    if n_page.status_code != 200:
        time.sleep(2)
        n_page = requests.get(url+str(i))

    n_finder = BeautifulSoup(n_page.text, 'lxml')
    try:
        content = n_finder.find_all('div', {'class': ['articulo-texto']})
        for div in content:
            n_url = div.findChild('h2').findChild()['href']
            links.append(base_url + n_url)

    except Exception as e:
        traceback += f'{i}. Error: {e}\n'

titles = []
texts = []
dates = []
pre_titles = []

traceback += 'Contenido:\n'
for link in links:
    k_page = requests.get(link)
    if k_page.status_code != 200:
        time.sleep(2)
        k_page = requests.get(link)

    k_finder = BeautifulSoup(k_page.text, 'lxml')
    try:
        # Obtiene tipo de nota
        pre_title = k_finder.find(
            'div', {'class': ['articulo-antetitulo']}).findChild().getText()
        pre_titles.append(pre_title)
        # Obtiene texto
        text = ''
        for p in k_finder.find_all('p'):
            text += p.getText()
        texts.append(text)
        # Obtiene titulo
        title = k_finder.find('h1', {'class': ['articulo-titulo']}).getText()
        titles.append(title)
        # Obtiene fecha
        date = k_finder.find(
            'a', {'title': ['Ver todas las noticias de esta fecha']})['href']
        dates.append(date)
        
    except Exception as e:
        traceback += f'{link}. Error: {e}\n'

try:
    data = {
        'Titulo': titles,
        'Texto': texts,
        'Fecha': dates,
        'Tipo': pre_titles
    }
    df = pd.DataFrame(data)
    df.to_csv('news_data.csv', index=False)
    print('Operación completa.')
except Exception as e:
    traceback += f'{link}. Error: {e}\n'




with open('traceback.txt', 'w', encoding='UTF-8') as file:
    file.write(traceback)

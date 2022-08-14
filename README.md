---
jupyter:
  kernelspec:
    display_name: Python 3.8.5 (\'base\')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.5
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: bf2d4fae2dd7f9d0a1cb92fe5a57ae32bb8cac3d6e2203bb2bd58a3e2ed832c0
---

# Estructura del repositorio
>Data: Contiene el script que se usó para obtener el set de datos y el archivo de datos.

>Hoja de análisis: !Es del desarrollo del análisis!

# Descripción
Con el fin de dar una aproximación alternativa a las series de tiempo y el precio de las acciones  de la compañía Tesla Motors se analizará la relación entre las noticias publicadas en la página 'El País' sobre la compañía y el precio de la misma. Se aplicará técnicas de procesamiento de lenguaje natural y se mostrará la correlación para obtener un modelo de clasificación.

Enlace: https://cincodias.elpais.com/tag/tesla_motors/
# Desarrollo
Cómo punto de partida tenemos los siguientes datos
<img src='reporte\serie.png'>
Observamos que entre las notas existen diferentes tipos, es fácil decir qué tipo de notas predomina.
<img src='reporte\distribucion.png'>
Luego, cómo aproximación a una posible relación entre el texto y el impacto, se clasifican los datos en dos grupos: aquellos cuya diferencia de cierre y apertura es positiva o negativa.

Del proceso anterior es posible visualizar que palabras son más frecuentes en cada caso
## Caso de cambio positivo

<img src='reporte\frases1.png'>

## Caso de cambio negativo

<img src='reporte\frases2.png'>

No se observan diferencias entre la frecuencia de las palabras, esto da pie a dos sub-casos: la relación se da en palabras menos comúnes, o se requiere de contexto, el cuál podría realizarse mediante n-gramas.

# Conclusión
A pesar de no observarse una relación significativa entre el texto y el cambio en el valor de las acciones, al momento de implementarse los modelos predictivos, se presentó una variación grande en las puntuaciones de rendimiento de los modelos con diferentes segmentaciones del conjunto de datos; esto da pie a que se pueda hallar una relación fuerte entre palabras clave si se trabaja con un conjunto más grande de datos.

Hay que tener en cuenta que este proyecto no es dirigido a generar un modelo que prediga el cambio en las acciones, pues está orientado a observar una posible relación entre las noticias y el cambio; ya que el orden de la obtención de los datos en un entorno real es diferente al realizado aquí.
 

var net;
var count = 0;

var valARTC = null;
var palabras = " ";
var validacion = 0;

const webcamE1 = document.getElementById("webcam");
const descCam1 = document.getElementById("console");

const clasificador = knnClassifier.create();
var webcam;

async function app(){
  net = await mobilenet.load();

  fetch('datos.json')
  .then(respuesta => respuesta.json())
  .then(usuarios => {
    Object.keys(usuarios).forEach((key) => {
      usuarios[key] = tf.tensor(usuarios[key], [usuarios[key].length / 1024, 1024]);
      console.log(key);
    });
    clasificador.setClassifierDataset(usuarios);
  })


  webcam = await tf.data.webcam(webcamE1);

  while(true){
    const img = await webcam.capture();
    const result = await net.classify(img);
    const activation = net.infer(img,"conv_preds");
    var result2;

    try{
      result2 = await clasificador.predictClass(activation);
    }catch(error){
      result2 = {}
    }

    const clases = ["No entrenado","cable","monitor","data","laptop","arduino","computadora","sensor","control"];

    descCam1.innerHTML = "<h4>PREDICCION</h4><p>" + result[0].className +
    "</p><h4>PROBABILIDAD</h4><p>" + result[0].probability + "</p>";

    if(valARTC == clases[result2.label]){

      validacion = validacion + 1;

    }else{
      validacion = 0;
    }

    valARTC = clases[result2.label];

    if(validacion==50){
      $("#enlace").get(0).click();
    }
    
    try{

      document.getElementById('enlace').innerHTML=clases[result2.label];
      //document.getElementById('enlace').href="https://www.youtube.com/results?search_query="+clases[result2.label];
      
    }catch(error){
      document.getElementById("console2").innerHTML="No entrenado";
    }

    img.dispose();
    await tf.nextFrame();
  }

}

async function addExample(){

  var classId = $("#IdEntrenarObjeto").val();

  console.log("Ejemplo agregado");
  const img = await webcam.capture();
  const activation = net.infer(img,true);
  clasificador.addExample(activation, classId);

  console.log(clasificador);

  img.dispose();
}

async function CargarDATA(classifierModel){
  console.log("HOLA");
  fetch('datos.json')
  .then(respuesta => respuesta.json())
  .then(usuarios => {
    Object.keys(usuarios).forEach((key) => {
      usuarios[key] = tf.tensor(usuarios[key], [usuarios[key].length / 1024, 1024]);
      console.log(key);
    });
    classifierModel.setClassifierDataset(usuarios);
  })
}


async function guardarmodelo(){

  var datasets = await clasificador.getClassifierDataset();
  var datasetObject = {};
  var json_arr = {};
  Object.keys(datasets).forEach(async (key) => {
    let data = await datasets[key].dataSync();
    datasetObject[key] = Array.from(data);
  });
  
  for(var i=0;i<3;i++){
    json_arr[i] = await datasets[i];
  }

  console.log(datasetObject);
  console.log(json_arr);

  var jsonModel = JSON.stringify(datasetObject);
  console.log(jsonModel);

  var myJsonString = JSON.stringify(json_arr);
  console.log(myJsonString);

  let downloader = document.createElement('a');
  downloader.download = "datos.json";
  downloader.href = 'data:text/text;charset=utf-8,' + encodeURIComponent(jsonModel);
  document.body.appendChild(downloader);
  downloader.click();
  downloader.remove();

  palabras = " ";
  //var news = "lap";
 // window.location.href = 'http://localhost/iComputacional/estudiante/buscador/1/recientes/'+news;
}






app();
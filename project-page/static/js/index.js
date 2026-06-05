window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/images/attention/";
var INTERP_ACT = "fwd";
var NUM_INTERP_FRAMES = 3;
var SLIDER_POS = 0;

var interp_images = [];
function preloadInterpolationImages() {

  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {

    if (i == 0) {
      str = "img";
    } else if (i == 1) {
      str = "overlay";
    } else if (i == 2) {
      str = "att";
    }

    var path = INTERP_BASE + INTERP_ACT + '_' + str + '.png';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function () { return false; };
  image.oncontextmenu = function () { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function () {
  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");

  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  }

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach('.carousel', options);

  // Loop on each carousel initialized
  for (var i = 0; i < carousels.length; i++) {
    // Add listener to  event
    carousels[i].on('before:show', state => {
      console.log(state);
    });
  }

  // Access to bulmaCarousel instance of an element
  var element = document.querySelector('#my-element');
  if (element && element.bulmaCarousel) {
    // bulmaCarousel instance is available as element.bulmaCarousel
    element.bulmaCarousel.on('before-show', function (state) {
      console.log(state);
    });
  }

  /*var player = document.getElementById('interpolation-video');
  player.addEventListener('loadedmetadata', function() {
    $('#interpolation-slider').on('input', function(event) {
      console.log(this.value, player.duration);
      player.currentTime = player.duration / 100 * this.value;
    })
  }, false);*/
  // preloadInterpolationImages();

  // $('#interpolation-slider').on('input', function (event) {
  //   setInterpolationImage(this.value);
  // });
  // setInterpolationImage(0);
  // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();

})

function appendImage(path, dst) {
  var image = new Image();

  image.onload = function () {
    $(dst).empty().append(image);
  }

  image.src = path;
  image.ondragstart = function () { return false; };
  image.oncontextmenu = function () { return false; };
}

var SAMPLE_POS = 0;
var DROP_MOD_BASE = "./static/images/navigation/inv_strip_"
function setDropMod(drop) {

  var buttons = ["drop-rgb-button", "drop-depth-button", "sample", "rgbd-button"];
  for (const x of buttons) {
    $(`#${x}`).removeClass('is-info');
  }

  if (drop == "sample") {
    SAMPLE_POS += 1;
    drop = "none";
  }

  var path = DROP_MOD_BASE + String(drop) + '_' + String(SAMPLE_POS % 3) + '.gif';
  appendImage(path, '#dropmod-wrapper-input');

  if (drop == "rgb") {
    $(`#drop-rgb-button`).addClass('is-info');
  } else if (drop == "depth") {
    $(`#drop-depth-button`).addClass('is-info');
  } else {
    $(`#rgbd-button`).addClass('is-info');
  }
}

function setAttMask(action) {

  INTERP_ACT = action

  var interp_images = [];
  preloadInterpolationImages();
  $('#interpolation-slider').on('input', function (event) {
    setInterpolationImage(this.value);
  });

  var type = $("#interpolation-image-wrapper").children()[0].src.split('/').slice(-1)[0].split("_").slice(-1)[0].split('.')[0];
  if (type == "img") {
    idx = 0;
  } else if (type == "overlay") {
    idx = 1;
  } else if (type == "att") {
    idx = 2;
  }

  setInterpolationImage(idx);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
  $(`#${action}-button`).addClass('is-info');
  if (action == "fwd") {
    $(`#left-button`).removeClass('is-info');
  } else if (action == "left") {
    $(`#fwd-button`).removeClass('is-info');
  }
}

// Improvement Videos Carousel Functionality
$(document).ready(function() {
  // Video data for the carousel
  const improvementVideos = [
    {
      src: "./static/videos/improvement/red_pentagon_blue_moon_improv.mp4",
      title: "Red Pentagon to Blue Moon"
    },
    {
      src: "./static/videos/improvement/green_cube_blue_moon_improv.mp4",
      title: "Green Cube to Blue Moon"
    },
    {
      src: "./static/videos/improvement/yellow_pentagon_red_moon_improv.mp4",
      title: "Yellow Pentagon to Red Moon"
    },
    {
      src: "./static/videos/improvement/yellow_star_blue_cube_improv.mp4",
      title: "Yellow Star to Blue Cube"
    },
    {
      src: "./static/videos/improvement/red_moon_green_star_improv.mp4",
      title: "Red Moon to Green Star"
    }
  ];

  let currentVideoIndex = 0;
  const videoElement = document.getElementById('improvementVideo');
  const taskTitle = document.getElementById('taskTitle');

  // Function to update video and info with smooth transition
  function updateCarousel(index) {
    const video = improvementVideos[index];
    
    // Add fade-out class for smooth transition
    videoElement.classList.add('fade-out');
    
    // Wait for fade-out to complete, then update content
    setTimeout(() => {
      videoElement.src = video.src;
      taskTitle.textContent = video.title;
      
      // Remove fade-out and add fade-in for smooth appearance
      videoElement.classList.remove('fade-out');
      videoElement.classList.add('fade-in');
      
      // Remove fade-in class after transition completes
      setTimeout(() => {
        videoElement.classList.remove('fade-in');
      }, 300);
    }, 150); // Half of the transition duration
  }

  // Navigation functions
  function nextVideo() {
    currentVideoIndex = (currentVideoIndex + 1) % improvementVideos.length;
    updateCarousel(currentVideoIndex);
  }

  function prevVideo() {
    currentVideoIndex = (currentVideoIndex - 1 + improvementVideos.length) % improvementVideos.length;
    updateCarousel(currentVideoIndex);
  }

  // Event listeners
  document.getElementById('nextBtn').addEventListener('click', nextVideo);
  document.getElementById('prevBtn').addEventListener('click', prevVideo);

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
      prevVideo();
    } else if (e.key === 'ArrowRight') {
      nextVideo();
    }
  });

  // Initialize carousel
  updateCarousel(0);
});
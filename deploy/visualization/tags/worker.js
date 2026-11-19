// Vendored browser sources from js-aruco2 2.0.0 (0491d5d228746411d0e9dd602b98f48636a644ba).
// https://github.com/damianofalcioni/js-aruco2
importScripts('js-aruco2/cv.js', 'js-aruco2/aruco.js', 'js-aruco2/apriltag_36h11.js');

// The field trial uses the first two codes. Require an exact bit match, as
// in the recording reviewer's OpenCV 36h11 dictionary (maxCorrectionBits=0).
const family = AR.DICTIONARIES.APRILTAG_36h11;
AR.DICTIONARIES.boxTags = {...family, codeList: family.codeList.slice(0, 2)};
const detector = new AR.Detector({dictionaryName: 'boxTags', maxHammingDistance: 0});
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext('2d', {willReadFrequently: true});

self.onmessage = ({data: bitmap}) => {
  const {width, height} = bitmap;
  try {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.drawImage(bitmap, 0, 0);
  } finally {
    bitmap.close();
  }
  const tags = detector.detect(context.getImageData(0, 0, width, height));
  self.postMessage({width, height, tags});
};

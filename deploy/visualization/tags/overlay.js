// The preview keeps refreshing during detection. A result belongs only to
// the image from which its bitmap was taken, including across reconnects.
class BoxTagOverlay {
  constructor(image, canvas, button, status) {
    this.image = image;
    this.canvas = canvas;
    this.status = status;
    this.button = button;
    this.frame = null;
    this.worker = null;
    this.pending = null;
    this.nextSample = 0;
    image.addEventListener('load', () => this.sample());
    button.onclick = () => this.worker ? this.stop() : this.start();
  }

  start() {
    try {
      const worker = new Worker('/tags/worker.js');
      this.worker = worker;
      this.button.setAttribute('aria-pressed', 'true');
      worker.onerror = () => {
        if (this.worker === worker) this.stop('标签识别不可用，请检查浏览器后重试');
      };
      worker.onmessage = ({data}) => {
        if (this.worker !== worker) return;
        const frame = this.pending;
        this.pending = null;
        if (frame && frame === this.frame) this.draw(data);
      };
      this.status.textContent = this.frame ? '等待识别' : '等待有效图像';
      this.sample();
    } catch {
      this.stop('当前浏览器无法启动标签识别');
    }
  }

  stop(message = '') {
    this.worker?.terminate();
    this.worker = null;
    this.pending = null;
    if (this.frame) this.frame.sampled = false;
    this.button.setAttribute('aria-pressed', 'false');
    this.clear();
    this.status.textContent = message;
  }

  clear() {
    this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  update(jpeg, session) {
    if (!jpeg && !this.frame) return;
    if (jpeg !== this.frame?.jpeg || session !== this.frame?.session) {
      this.frame = jpeg ? {jpeg, session, sampled: false} : null;
      if (this.worker) {
        this.clear();
        this.status.textContent = jpeg ? '等待识别' : '等待有效图像';
      }
    }
    this.sample();
  }

  async sample() {
    const worker = this.worker, frame = this.frame;
    if (!worker || !frame || frame.sampled || this.pending || document.hidden ||
        !this.image.complete || !this.image.naturalWidth || performance.now() < this.nextSample) return;
    this.pending = frame;
    frame.sampled = true;
    this.nextSample = performance.now() + 500;
    try {
      const bitmap = await createImageBitmap(this.image);
      if (this.worker === worker && this.frame === frame) {
        worker.postMessage(bitmap, [bitmap]);
      } else {
        bitmap.close();
        if (this.worker === worker) this.pending = null;
      }
    } catch {
      if (this.worker === worker) this.stop('无法读取标签识别图像，可点击重试');
    }
  }

  draw({width, height, tags}) {
    const canvas = this.canvas;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const fontSize = Math.max(14, height / 15);
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.lineJoin = 'round';
    for (const {id, corners} of tags) {
      ctx.lineWidth = Math.max(2, height / 110);
      ctx.strokeStyle = '#00e98b';
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      for (const point of corners.slice(1)) ctx.lineTo(point.x, point.y);
      ctx.closePath();
      ctx.stroke();
      const label = 'ID ' + id;
      const x = Math.min(...corners.map(p => p.x), width - ctx.measureText(label).width);
      const y = Math.max(fontSize, Math.min(...corners.map(p => p.y)) - 4);
      ctx.strokeStyle = '#153527';
      ctx.strokeText(label, x, y);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x, y);
    }
    this.status.textContent = tags.length ? tags.map(tag => 'ID ' + tag.id).join(' · ') : '未检出 ID 0 或 1';
  }
}

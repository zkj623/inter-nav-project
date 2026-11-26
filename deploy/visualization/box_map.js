'use strict';

// The fitter reports the long axis in experiment coordinates, modulo 180°.
function boxFootprint(pose) {
  const [x, y] = pose.center || [];
  const [a, b] = pose.dimensions_m || [];
  const angle = pose.long_axis_rad ?? (a === b ? pose.face_angle_rad : null);
  if (![x, y, a, b, angle].every(Number.isFinite) || a <= 0 || b <= 0) return null;
  const length = Math.max(a, b) / 2, width = Math.min(a, b) / 2;
  const c = Math.cos(angle), s = Math.sin(angle);
  return [[-length,-width], [length,-width], [length,width], [-length,width]].map(([u,v]) =>
    [x + u*c - v*s, y + u*s + v*c]);
}

class BoxMapOverlay {
  constructor(canvas, onStatus) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onStatus = onStatus;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  update(frame, offline, requestStarted) {
    if (this.frame?.session_id !== frame?.session_id) this.fit = null;
    this.frame = frame;
    this.offline = offline;
    this.updateTracking(frame, requestStarted);
  }

  updateTracking(feed, requestStarted) {
    if (feed?.session_id !== this.frame?.session_id) return;
    const fit = feed?.box_window;
    if (fit && (!this.fit || fit.as_of > this.fit.as_of)) {
      this.fit = fit;
      this.requestStarted = requestStarted;
      this.feedOffline = feed.finished || feed.session_age > 2;
    }
    this.draw();
  }

  draw() {
    clearTimeout(this.expiryTimer);
    const {canvas, ctx, frame} = this;
    const {width, height} = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
      canvas.width = Math.round(width*dpr);
      canvas.height = Math.round(height*dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    canvas.setAttribute('aria-label', '地图箱位');
    const map = frame?.map, fit = this.fit, layout = map?.projection;
    const now = fit ? fit.as_of + (performance.now()-this.requestStarted)/1000 : 0;
    const offline = this.offline || this.feedOffline;
    const stale = offline || map?.stale || (!frame?.box_feed && frame?.performance?.input_stale) || fit?.error;
    this.onStatus?.(fit, now, stale);
    if (!width || !height || !map?.ready || !layout || !fit || map.frame !== 'experiment'
        || (fit.frame && fit.frame !== map.frame)) return;

    // Match object-fit: contain, including letterboxing on resize.
    const [imageWidth, imageHeight] = layout.image_size;
    const scale = Math.min(width/imageWidth, height/imageHeight);
    const [left, top, w, h] = layout.viewport;
    const x0 = (width-imageWidth*scale)/2 + left*scale;
    const y0 = (height-imageHeight*scale)/2 + top*scale;
    const [xmin,xmax] = layout.x_limits, [ymin,ymax] = layout.y_limits;
    const pixel = ([x,y]) => [x0+(x-xmin)/(xmax-xmin)*(w-1)*scale,
                              y0+(ymax-y)/(ymax-ymin)*(h-1)*scale];
    const current = new Map(fit.detections.map(row => [row.object_id, row]));
    // The session retains the last position confirmed while its evidence was fresh.
    const poses = Object.entries(fit.last_seen || {});
    const labels = [];
    let nextExpiry = Infinity;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w*scale, h*scale);
    ctx.clip();
    for (const [id, pose] of poses) {
      const corners = boxFootprint(pose);
      if (!corners) continue;
      const row = current.get(id);
      const fresh = !stale && row?.valid
        && row.captured_at <= now && now < row.expires_at;
      if (fresh) nextExpiry = Math.min(nextExpiry, row.expires_at);
      const label = '箱位 ' + id;
      labels.push(label);
      const points = corners.map(pixel);
      const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
      const right = x0+w*scale, bottom = y0+h*scale;
      if (Math.max(...xs)<x0 || Math.min(...xs)>right || Math.max(...ys)<y0 || Math.min(...ys)>bottom) continue;
      const color = '#006e63';
      ctx.beginPath();
      points.forEach(([x,y], i) => i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
      ctx.closePath();
      ctx.fillStyle = 'rgba(0,151,130,.25)';
      ctx.fill();
      ctx.strokeStyle = 'white'; ctx.lineWidth = 4; ctx.stroke();
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '600 12px system-ui, sans-serif';
      const labelWidth = ctx.measureText(label).width+10;
      const lx = Math.max(x0, Math.min(right-labelWidth, Math.min(...xs)));
      let ly = Math.min(...ys)-21;
      if (ly < y0) ly = Math.min(bottom-18, Math.max(...ys)+3);
      ctx.fillStyle = color; ctx.fillRect(lx, ly, labelWidth, 18);
      ctx.fillStyle = 'white'; ctx.textBaseline = 'middle'; ctx.fillText(label, lx+5, ly+9);
    }
    ctx.restore();
    if (labels.length) canvas.setAttribute('aria-label', labels.join('，'));
    // Refresh the scan age independently of the persistent map outline.
    if (Number.isFinite(nextExpiry)) {
      this.expiryTimer = setTimeout(() => this.draw(), Math.max(1, (nextExpiry-now)*1000));
    }
  }
}

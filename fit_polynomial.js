{
let canvas = document.querySelector("#polynomial");
let ctx = canvas.getContext("2d");
let px = [], py = [];
let clicked = false;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

let coefs = [];
for (let i = 0; i < 4; i++)
	coefs.push( tf.variable(tf.scalar(Math.random())) );

canvas.addEventListener('mousedown', () => { clicked = true; });
canvas.addEventListener('mouseup', () => { clicked = false; fit(); });

canvas.addEventListener('mousemove', (e) => {
	if (clicked) {
		let nx = e.offsetX / 250 - 1;
		let ny = e.offsetY / 250 - 1;
		if (px.length)
		{
			let i = px.length - 1;
			if (Math.sqrt(Math.pow(nx - px[i], 2) + Math.pow(ny - py[i], 2)) < 0.2)
				return;
		}
		px.push(nx);
		py.push(ny);
		draw();
	}
});

function predict(x)
{
	return tf.tidy(() => {
		x = tf.tensor(x);
		let res = coefs[0];
		let x_p = x;
		for (let i = 1; i < coefs.length; i++)
		{
			res = res.add(x_p.mul(coefs[i]));
			x_p = x_p.mul(x);
		}
		return res;
	});
}

function loss(estimate, target)
{
	return tf.tidy(() => {
		return estimate.sub(tf.tensor(target)).square().mean();
	});
}

function draw(lineColor)
{
	ctx.clearRect(0, 0, canvas.width, canvas.height);

	ctx.fillStyle = 'rgb(0, 0, 0)';
	for (let i = 0; i < px.length; i++)
	{
		ctx.beginPath();
		ctx.arc((px[i]+1) * 250, (py[i]+1) * 250, 5, 0, 2*Math.PI);
		ctx.fill();
	}

	let x = [];
	for (let i = -1 - 0.05; i <= 1 + 0.05; i += 0.05)
		x.push(i);
		
	ctx.strokeStyle = lineColor || 'rgb(0, 0, 0)';
	ctx.lineWidth = 3;
	ctx.beginPath();
	tf.tidy(() => {
		let y = predict(x).dataSync();
		for (let i = 0; i < x.length; i++)
			ctx.lineTo((x[i]+1) * 250, (y[i]+1) * 250);
	});
	ctx.stroke();
}

function fit()
{
	if (!px.length)
		return;

	let res = optimizer.minimize(() => {
		return loss(predict(px), py);
	}, true);

	if (res.dataSync() > 0.005)
	{
		draw();
		setTimeout(fit, 10);
	}
	else
		draw('rgb(0, 200, 0');
}

document.querySelector('#clearPoly').addEventListener('click', () => {
	px = [];
	py = [];
	draw();
})
}
{
let canvas = document.querySelector("#model");
let ctx = canvas.getContext("2d");
let px = [], py = [];
let clicked = false;
let training = false;

let model = tf.sequential();
model.add(tf.layers.dense({
	inputShape: [1],
	units: 10,
	activation: 'sigmoid'
}));
model.add(tf.layers.dropout({
	rate: 0.2
}));
model.add(tf.layers.dense({
	units: 10,
	activation: 'sigmoid'
}));
model.add(tf.layers.dense({
	units: 1
}));

model.compile({
	optimizer: tf.train.adam(0.2),
	loss: 'meanSquaredError'
})

canvas.addEventListener('mousedown', () => { clicked = true; });
canvas.addEventListener('mouseup', () => { clicked = false; fit(); });

canvas.addEventListener('mousemove', (e) => {
	if (clicked) {
		let nx = e.offsetX / 250 - 1;
		let ny = e.offsetY / 250 - 1;
		if (px.length)
		{
			let i = px.length - 1;
			if (Math.sqrt(Math.pow(nx - px[i][0], 2) + Math.pow(ny - py[i][0], 2)) < 0.2)
				return;
		}
		px.push([nx]);
		py.push([ny]);
		draw();
	}
});

function predict(x)
{
	return tf.tidy(() => { return model.predict(tf.tensor(x)) });
}

function draw(lineColor)
{
	if (training)
		return;

	ctx.clearRect(0, 0, canvas.width, canvas.height);

	ctx.fillStyle = 'rgb(0, 0, 0)';
	for (let i = 0; i < px.length; i++)
	{
		ctx.beginPath();
		ctx.arc((px[i][0]+1) * 250, (py[i][0]+1) * 250, 5, 0, 2*Math.PI);
		ctx.fill();
	}

	let x = [];
	for (let i = -1 - 0.05; i <= 1 + 0.05; i += 0.05)
		x.push([i]);
	
	let yt = predict(x), y = yt.dataSync();
	yt.dispose();

	ctx.strokeStyle = lineColor || 'rgb(0, 0, 0)';
	ctx.lineWidth = 3;
	ctx.beginPath();
	for (let i = 0; i < x.length; i++)
		ctx.lineTo((x[i][0]+1) * 250, (y[i]+1) * 250);
	ctx.stroke();
}

async function fit()
{
	if (!px.length || training)
		return;

	let epochs = 2;
	training = true;
	
	let x = tf.tensor(px), y = tf.tensor(py);
	const h = await model.fit(x, y, {
		shuffle: true,
		epochs
	});
	x.dispose();
	y.dispose();

	training = false;

	if (h.history.loss[0] > 0.02)
	{
		draw();
		setTimeout(fit, 10);
	}
	else
		draw('rgb(0, 200, 0');

}

document.querySelector('#clearModel').addEventListener('click', () => {
	px = [];
	py = [];
	draw();
})
}
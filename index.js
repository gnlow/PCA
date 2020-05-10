const {PCA} = require("ml-pca")
const gaussian = require("gaussian")
const dataset = require("@randkid/size")
const d = dataset.f_dropna.data.splice(0, 5)
const pca = new PCA(d)
const distributions = 
    pca.getStandardDeviations()
        .map(σ => gaussian(0, σ**2))
console.log( pca.invert([distributions.map(d => d.ppf(Math.random()))]) )
const {PCA} = require("ml-pca")
const gaussian = require("gaussian")
const dataset = require("@randkid/size")
const {groupBy} = require("lodash")

const ds = groupBy( dataset.f_dropna.data, 0 )
for(const k in ds){
    const pca = new PCA(ds[k])
    console.log(pca.getStandardDeviations()[0])
    const distributions = 
        pca.getStandardDeviations()
            .splice(0, 5)
            .map(σ => gaussian(0, σ**2))
    console.log( pca.invert([distributions.map(d => d.ppf(Math.random()))]) )
}
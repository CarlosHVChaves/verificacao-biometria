//jshint esversion:6

function analyzeStuff() {

    let success = result => {

        $('#analyze-photo-btn').button('reset');

        let scoreArray = result.split("\n");

        let html = "";

        scoreArray.forEach(score => {

            //skip invalid results
            if ("" === score) return;

            (score < 1) ?
            //if score is less than 1, green
            html += `<div class="col-md-1 green">${score}<br>Same person</div>`
                //otherwise, red
                : html += `<div class="col-md-1 red">${score}<br>Different person</div>`;
        });

        $('#result').html(html);

    };

    let error = error => {
        console.error("Error while analyzing images: ", error);
    };

    let ajaxSettings = {
        type: "GET",
        url: '/test',
        success: success,
        error: error,
        statusCode: {
            404: function() {
                reject("page not found");
            }
        }
    };

    $.ajax(ajaxSettings);
}

function firstRowItems(string) {
    return (string.indexOf('first') !== -1);
}

function secondRowItems(string) {
    return (string.indexOf('second') !== -1);
}

function analyzeStuff2() {

    let success = result => {


        let scoreArray = result.output.split("\n");

        $('#analyze-video-photo-btn').button('reset');

        let html = "";

        result.fileList.filter(firstRowItems).forEach((file, index) => {

            if (index >= scoreArray.length) return;
            console.log("updating " + file + "on index " + index);
            let $img = $("<img>", {
                src: '/getImage?file=' + file,
                width: '80px',
                height: '120px'
            });

            $("#video-image-results .video-image-result:nth-child(" + index + ")").append($img);
        });

        scoreArray.forEach((score, index) => {
            if (index >= result.fileList.filter(firstRowItems).length) return;

            //skip invalid results
            if ("" === score) return;

            html = "<br><br>";

            if (score < 1) {
                //if score is less than 1, green
                html += `<div class="green">${score}<br>Same person</div>`;
            } else {
                //otherwise, red
                html += `<div class="red">${score}<br>Different person</div>`;
            }

            html += "<br>";

            $("#video-image-results .video-image-result:nth-child(" + index + ")").append(html);
        });

        result.fileList.filter(secondRowItems).forEach((file, index) => {
            if (index >= result.fileList.filter(firstRowItems).length) return;
            let $img = $("<img>", {
                src: '/getImage?file=' + file,
                width: '80px',
                height: '120px'
            });
            console.log("updating " + file + "on index " + index);
            $("#video-image-results .video-image-result:nth-child(" + index + ")").append("<br>");
            $("#video-image-results .video-image-result:nth-child(" + index + ")").append($img);
        });

    };

    let error = error => {
        console.error("Error while analyzing images: ", error);
    };

    let ajaxSettings = {
        type: "GET",
        url: '/testVideo',
        success: success,
        error: error
    };

    $.ajax(ajaxSettings);
}

$(document).ready(() => {


    $('#analyze-photo-btn').on('click', function() {
        analyzeStuff();
        var $this = $(this);
        $this.button('loading');

    });

    $('#analyze-video-photo-btn').on('click', function() {
        analyzeStuff2();
        var $this = $(this);
        $this.button('loading');

    });

}); 
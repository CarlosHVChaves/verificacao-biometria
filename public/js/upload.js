/**
 * Created by remi on 17/01/15.
 */
 (function () {

    var uploadfiles = document.querySelector('#uploadfiles1');
    uploadfiles.addEventListener('change', function () {
        var files = this.files;
        for(var i=0; i<files.length; i++){
            uploadFile1(this.files[i]);
        }

    }, false);

    var uploadfiles2 = document.querySelector('#uploadfiles2');
    uploadfiles2.addEventListener('change', function () {
        var files = this.files;
        for(var i=0; i<files.length; i++){
            uploadFile2(this.files[i]);
        }

    }, false);

    /**
     * Upload a file
     * @param file
     */
    function uploadFile1(file){
        var url = "/upload_file";
        var xhr = new XMLHttpRequest();
        var fd = new FormData();
        xhr.open("POST", url, true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // Every thing ok, file uploaded
                console.log(xhr.responseText); // handle response.
            }
        };
        fd.append('file', file);
        xhr.send(fd);
    }

    function uploadFile2(file){
        var url = "/file-upload2";
        var xhr = new XMLHttpRequest();
        var fd = new FormData();
        xhr.open("POST", url, true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // Every thing ok, file uploaded
                console.log(xhr.responseText); // handle response.
            }
        };
        fd.append('file', file);
        xhr.send(fd);
    }
}());

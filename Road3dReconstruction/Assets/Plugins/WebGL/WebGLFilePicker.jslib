// Bridge that lets the Unity WebGL build trigger a native browser file picker
// and hand the resulting Blob URL back to managed code. The C# side receives
// a JSON-encoded payload via SendMessage so the standard UnityWebRequest path
// can fetch the bytes from the blob URL — identical to how editor builds work
// with a regular file path.
mergeInto(LibraryManager.library, {
  // accept: e.g. "video/*,image/*". empty → all.
  // gameObject / methodName receive a JSON string when the user selects a file.
  WebGLFilePicker_Open: function (gameObjectPtr, methodNamePtr, acceptPtr) {
    var gameObject = UTF8ToString(gameObjectPtr);
    var methodName = UTF8ToString(methodNamePtr);
    var accept = UTF8ToString(acceptPtr) || "";

    // Reuse the same hidden input across calls so we don't leak DOM nodes.
    var input = document.getElementById("__unity_file_picker__");
    if (!input) {
      input = document.createElement("input");
      input.type = "file";
      input.id = "__unity_file_picker__";
      input.style.display = "none";
      document.body.appendChild(input);
    }
    input.value = "";
    input.accept = accept;

    input.onchange = function (event) {
      var file = event.target.files && event.target.files[0];
      if (!file) {
        SendMessage(gameObject, methodName, JSON.stringify({ cancelled: true }));
        return;
      }
      var url = URL.createObjectURL(file);
      var payload = {
        url: url,
        name: file.name,
        size: file.size,
        type: file.type || ""
      };
      SendMessage(gameObject, methodName, JSON.stringify(payload));
    };
    // .click() must be inside a user gesture (the call originates from a Unity
    // button click, so this works). Calling it from non-gesture contexts (e.g.
    // Start()) silently fails — that's a browser policy, not a bug here.
    input.click();
  },

  // Releases the object URL after the upload finishes so the browser can GC the blob.
  WebGLFilePicker_Revoke: function (urlPtr) {
    var url = UTF8ToString(urlPtr);
    if (url) {
      try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
    }
  }
});

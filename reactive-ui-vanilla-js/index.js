function store(data = {}, name = "optsys") {
    /**
     * Emit a custom event.
     * @param {string} type The event type.
     * @param {any} detail Any details to pass along with the event.
     * @returns {void}
     */
    function emit(type, detail) {
        let event = new CustomEvent(type, { 
            bubbles: true,
            cancelable: true,
            detail: detail,
         });
        
         return document.dispatchEvent(event);
    }

    // The Proxy object allows you to create an object that can be used in place of the original
    // object, but which may redefine fundamental Object operations like getting, setting, and
    // defining properties.
    return new Proxy(data, {
        get: function (obj, prop) {
            return obj[prop];
        },
        set: function (obj, prop, value) {
            if (obj[prop] === value) return true;
            obj[prop] = value;
            return true;
        },
        deleteProperty: function (obj, prop) {
            delete obj[prop];
            return true;
        }
    });
}
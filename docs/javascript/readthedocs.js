document.addEventListener("DOMContentLoaded", function(event) {
    // Use ReadTheDocs search addon instead of Material theme default
    document.querySelector(".md-search__input").addEventListener("focus", (e) => {
        const event = new CustomEvent("readthedocs-search-show");
        document.dispatchEvent(event);
    });
});

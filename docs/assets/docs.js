/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * What the documentation pages (docs/quickstart.html, docs/list_of_commands.html)
 * share: the theme and its remembered choice, the sidebar, and the sidebar's
 * highlight of the section being read. Each page links this file in its head,
 * so the theme is set before the page is drawn and a dark page never shows
 * light first. A page's own search and copy buttons stay in the page.
 */
(function () {
    "use strict";

    var KEY = "emapssn_docs_theme";
    var root = document.documentElement;
    var COLORS = { light: "#ffffff", dark: "#18181b" };

    // localStorage can be missing or throw (private windows, blocked site data).
    function readTheme() {
        try {
            var value = localStorage.getItem(KEY);
            return value === "dark" || value === "light" ? value : null;
        } catch (error) {
            return null;
        }
    }

    function writeTheme(theme) {
        try { localStorage.setItem(KEY, theme); } catch (error) { /* the choice just isn't remembered */ }
    }

    function systemTheme() {
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }

    function currentTheme() {
        return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    }

    // The button shows the theme a click switches to, as the Viewer's pages do.
    function syncThemeButton() {
        var button = document.getElementById("theme-toggle-btn");
        if (!button) return;
        var dark = currentTheme() === "dark";
        var label = button.querySelector(".theme-label");
        if (label) {
            label.textContent = (dark ? button.getAttribute("data-label-light") : button.getAttribute("data-label-dark")) || label.textContent;
        }
        var use = button.querySelector("use");
        if (use) use.setAttribute("href", dark ? "#i-sun" : "#i-moon");
        button.setAttribute("aria-label", label ? label.textContent : "");
    }

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        var metas = document.querySelectorAll('meta[name="theme-color"]');
        for (var i = 0; i < metas.length; i++) {
            metas[i].removeAttribute("media");
            metas[i].setAttribute("content", COLORS[theme]);
        }
        syncThemeButton();
    }

    applyTheme(readTheme() || systemTheme());

    // The theme chosen in the other documentation page applies here at once.
    window.addEventListener("storage", function (event) {
        if (event.key === KEY && (event.newValue === "dark" || event.newValue === "light")) {
            applyTheme(event.newValue);
        }
    });

    // Until a choice is made, the system's theme is followed.
    if (window.matchMedia) {
        var query = window.matchMedia("(prefers-color-scheme: dark)");
        var follow = function (event) { if (!readTheme()) applyTheme(event.matches ? "dark" : "light"); };
        if (query.addEventListener) query.addEventListener("change", follow);
        else if (query.addListener) query.addListener(follow);
    }

    function initTheme() {
        syncThemeButton();
        var button = document.getElementById("theme-toggle-btn");
        if (!button) return;
        button.addEventListener("click", function () {
            var next = currentTheme() === "dark" ? "light" : "dark";
            applyTheme(next);
            writeTheme(next);
        });
    }

    // On a wide screen the sidebar collapses; on a narrow one it is a drawer.
    function initSidebar() {
        var button = document.getElementById("toggle-sidebar-btn");
        var sidebar = document.getElementById("sidebar");
        if (!button || !sidebar) return;
        var narrow = window.matchMedia("(max-width: 900px)");
        var scrim = document.createElement("div");
        scrim.className = "sidebar-scrim";
        document.body.appendChild(scrim);

        function isOpen() {
            return narrow.matches
                ? document.body.classList.contains("sidebar-open")
                : !document.body.classList.contains("sidebar-collapsed");
        }

        function setOpen(open) {
            if (narrow.matches) document.body.classList.toggle("sidebar-open", open);
            else document.body.classList.toggle("sidebar-collapsed", !open);
            button.setAttribute("aria-expanded", open ? "true" : "false");
        }

        button.addEventListener("click", function () { setOpen(!isOpen()); });
        scrim.addEventListener("click", function () { setOpen(false); });
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && narrow.matches && isOpen()) setOpen(false);
        });
        sidebar.addEventListener("click", function (event) {
            if (narrow.matches && event.target.closest && event.target.closest(".menu-item a")) setOpen(false);
        });
        var reset = function () {
            document.body.classList.remove("sidebar-open");
            button.setAttribute("aria-expanded", isOpen() ? "true" : "false");
        };
        if (narrow.addEventListener) narrow.addEventListener("change", reset);
        else if (narrow.addListener) narrow.addListener(reset);
        reset();
    }

    // The menu entry of the section being read is highlighted. An entry's
    // data-target is the id of the element that starts its section.
    function initScrollSpy() {
        var items = Array.prototype.slice.call(document.querySelectorAll(".menu-item[data-target]"));
        if (!items.length) return;
        var menu = document.querySelector(".sidebar-menu");
        var targets = [];
        items.forEach(function (item) {
            var element = document.getElementById(item.getAttribute("data-target"));
            if (element) targets.push({ item: item, element: element });
        });
        if (!targets.length) return;
        targets.sort(function (a, b) {
            return a.element.compareDocumentPosition(b.element) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
        });
        var lockedUntil = 0;
        var queued = false;

        function show(item) {
            items.forEach(function (other) { other.classList.toggle("active", other === item); });
            if (menu && item.offsetParent) {
                var top = item.offsetTop - menu.offsetTop;
                if (top < menu.scrollTop + 8) menu.scrollTop = Math.max(0, top - 8);
                else if (top + item.offsetHeight > menu.scrollTop + menu.clientHeight - 8) {
                    menu.scrollTop = top + item.offsetHeight - menu.clientHeight + 8;
                }
            }
        }

        function update() {
            queued = false;
            if (Date.now() < lockedUntil) return;
            var line = 120;
            var current = targets[0];
            if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 60) {
                current = targets[targets.length - 1];
            } else {
                for (var i = 0; i < targets.length; i++) {
                    if (targets[i].element.getBoundingClientRect().top <= line) current = targets[i];
                    else break;
                }
            }
            show(current.item);
        }

        function schedule() {
            if (queued) return;
            queued = true;
            window.requestAnimationFrame(update);
        }

        // A click moves the highlight at once and leaves it while the page scrolls there.
        items.forEach(function (item) {
            var link = item.querySelector("a");
            if (!link) return;
            link.addEventListener("click", function () {
                show(item);
                lockedUntil = Date.now() + 800;
                window.setTimeout(schedule, 850);
            });
        });

        window.addEventListener("scroll", schedule, { passive: true });
        window.addEventListener("resize", schedule);
        update();
    }

    document.addEventListener("DOMContentLoaded", function () {
        initTheme();
        initSidebar();
        initScrollSpy();
    });
})();

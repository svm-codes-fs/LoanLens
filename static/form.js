// ------------------ STEP NAVIGATION ------------------

function nextStep() {

    const step1 = document.getElementById("step1");
    const inputs = step1.querySelectorAll("select, input");
    const errorMsg = document.getElementById("errorMsg");

    let isValid = true;

    inputs.forEach(input => {
        if (input.hasAttribute("required") && input.value === "") {
            isValid = false;
        }
    });

    if (!isValid) {
        errorMsg.style.display = "block";
        return;
    }

    // hide error if valid
    errorMsg.style.display = "none";

    // go to next step
    step1.classList.remove("active");
    document.getElementById("step2").classList.add("active");
}
function prevStep() {
    document.getElementById("step2").classList.remove("active");
    document.getElementById("step1").classList.add("active");
}

// ------------------ FORMAT FUNCTIONS ------------------

function formatINR(num) {
    if (num === "" || num === null || num === undefined) return "";
    return Number(num).toLocaleString("en-IN");
}
function cleanNumber(value) {
    return value.replace(/,/g, "");
}

// ------------------ GRADIENT ------------------

function setGradient(slider) {
    let percent = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;

    slider.style.background = `linear-gradient(to right,
        #004b8e 0%,
        #004b8e ${percent}%,
        #dcdcdc ${percent}%,
        #dcdcdc 100%)`;
}

// ------------------ SYNC FUNCTION ------------------
function sync(slider) {

    const wrapper = slider.closest(".slider-group");
    const numberInput = wrapper.querySelector(".calculator-num-inp");

    // ✅ Always format
    numberInput.value = formatINR(slider.value);

    setGradient(slider);
}
// ------------------ INIT ------------------
document.addEventListener("DOMContentLoaded", function () {

    const sliders = document.querySelectorAll(".range-input");

    sliders.forEach(slider => {

        const wrapper = slider.closest(".slider-group");
        const numberInput = wrapper.querySelector(".calculator-num-inp");

        // ✅ SET INITIAL VALUE PROPERLY
        numberInput.value = formatINR(slider.value);

        // ✅ SET INITIAL GRADIENT
        setGradient(slider);

        // ✅ EVENT LISTENER
        slider.addEventListener("input", () => {
            sync(slider);
        });

        // ✅ INPUT CHANGE
        numberInput.addEventListener("input", () => {

            let raw = numberInput.value;

            if (raw === "") return;

            let val = parseInt(cleanNumber(raw));

            if (isNaN(val)) return;

            if (val < slider.min) val = slider.min;
            if (val > slider.max) val = slider.max;

            slider.value = val;
            numberInput.value = formatINR(val);

            setGradient(slider);
        });

    });

});
function goBackToForm() {
    document.getElementById("step3").classList.remove("active");
    document.getElementById("step1").classList.add("active");
}
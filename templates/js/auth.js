const Auth = {
  ROLES: {
    admin: { label: "Admin", permissions: ["Manage users", "Billing & plans", "Full analytics", "System settings"] },
    client: { label: "Client", permissions: ["View own dashboard", "Manage resources", "Usage reports", "Support tickets"] },
    developer: { label: "Developer", permissions: ["API access", "Deploy instances", "View logs", "Test environments"] },
    analyst: { label: "Analyst", permissions: ["Read-only analytics", "Export reports", "Trend views"] },
  },

  async register({ name, email, password, role }) {
    return this.request("/register/", { name, email, password, role });
  },

  async login({ email, password, remember }) {
    return this.request("/login/", { email, password, remember });
  },

  async resetPassword(email, newPassword) {
    return this.request("/forgot-password/", { email, new_password: newPassword });
  },

  async logout() {
    return this.request("/logout/", {});
  },

  async currentUser() {
    const res = await this.request("/me/", {});
    return res.user || null;
  },

  async getUsers() {
    const res = await this.request("/users/", {});
    return res.users || [];
  },

  async uploadPhoto(file) {
    const form = new FormData();
    form.append("photo", file);
    const res = await fetch("/profile/photo/", {
      method: "POST",
      headers: { "X-CSRFToken": getCookie("csrftoken") },
      body: form,
    });
    return res.json();
  },

  async updateProfile({ name, email }) {
    return this.request("/profile/edit/", { name, email });
  },

  async request(path, data) {
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify(data),
      });
      return await res.json();
    } catch {
      return { ok: false, message: "Could not reach the server. Please try again." };
    }
  },
};

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
  return null;
}

function getField(id) {
  return document.getElementById(id);
}

function showError(input, message) {
  input.classList.add("error");
  const err = document.querySelector(`[data-error-for="${input.id}"]`);
  if (err) {
    err.textContent = message;
    err.classList.add("show");
  }
}

function clearError(input) {
  input.classList.remove("error");
  const err = document.querySelector(`[data-error-for="${input.id}"]`);
  if (err) err.classList.remove("show");
}

function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email);
}

function validatePassword(pw) {
  return pw.length >= 8;
}

function showAlert(id, message, type) {
  const el = getField(id);
  if (!el) return;
  el.textContent = message;
  el.className = "alert show " + (type === "success" ? "alert-success" : "alert-error");
  setTimeout(() => {
    el.className = "alert";
  }, 6000);
}

function attachPasswordToggles(container) {
  container.querySelectorAll("[data-toggle-password]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-toggle-password");
      const input = document.getElementById(targetId);
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btn.textContent = isPassword ? "🙈" : "👁";
    });
  });
}

const Auth = {
  DB_KEY: "cloudcount_users",
  SESSION_KEY: "cloudcount_session",
  REMEMBER_KEY: "cloudcount_remember",

  ROLES: {
    admin: { label: "Admin", permissions: ["Manage users", "Billing & plans", "Full analytics", "System settings"] },
    client: { label: "Client", permissions: ["View own dashboard", "Manage resources", "Usage reports", "Support tickets"] },
    developer: { label: "Developer", permissions: ["API access", "Deploy instances", "View logs", "Test environments"] },
    analyst: { label: "Analyst", permissions: ["Read-only analytics", "Export reports", "Trend views"] },
  },

  getUsers() {
    try {
      return JSON.parse(localStorage.getItem(this.DB_KEY)) || [];
    } catch {
      return [];
    }
  },

  saveUsers(users) {
    localStorage.setItem(this.DB_KEY, JSON.stringify(users));
  },

  findByEmail(email) {
    return this.getUsers().find((u) => u.email === email);
  },

  register({ name, email, password, role }) {
    const users = this.getUsers();
    const exists = users.some((u) => u.email === email);
    if (exists) {
      return { ok: false, message: "An account with this email already exists. Try logging in instead." };
    }
    const user = {
      id: "u_" + Date.now().toString(36),
      name: name.trim(),
      email: email,
      password: btoa(password),
      role: role,
      createdAt: new Date().toISOString(),
    };
    users.push(user);
    this.saveUsers(users);
    return { ok: true, user };
  },

  login({ email, password, remember }) {
    const user = this.findByEmail(email);
    if (!user) {
      return { ok: false, message: "No account found for this email. Please register first." };
    }
    if (user.password !== btoa(password)) {
      return { ok: false, message: "Incorrect password. Please try again." };
    }
    const session = { id: user.id, role: user.role, loginAt: new Date().toISOString() };
    const storage = remember ? localStorage : sessionStorage;
    storage.setItem(this.SESSION_KEY, JSON.stringify(session));
    localStorage.removeItem(this.REMEMBER_KEY);
    localStorage.setItem(this.REMEMBER_KEY, remember ? "yes" : "no");
    return { ok: true, user };
  },

  getSession() {
    try {
      const local = JSON.parse(sessionStorage.getItem(this.SESSION_KEY));
      if (local) return local;
      return JSON.parse(localStorage.getItem(this.SESSION_KEY));
    } catch {
      return null;
    }
  },

  currentUser() {
    const session = this.getSession();
    if (!session) return null;
    return this.getUsers().find((u) => u.id === session.id) || null;
  },

  logout() {
    sessionStorage.removeItem(this.SESSION_KEY);
    localStorage.removeItem(this.SESSION_KEY);
    localStorage.removeItem(this.REMEMBER_KEY);
  },

  resetPassword(email, newPassword) {
    const user = this.findByEmail(email);
    if (!user) {
      return { ok: false, message: "No account found for this email." };
    }
    const users = this.getUsers().map((u) =>
      u.email === email ? { ...u, password: btoa(newPassword) } : u
    );
    this.saveUsers(users);
    return { ok: true };
  },

  seedAdmin() {
    const users = this.getUsers();
    if (!users.some((u) => u.email === "admin@cloudcount.io")) {
      users.push({
        id: "u_admin",
        name: "System Admin",
        email: "admin@cloudcount.io",
        password: btoa("admin123"),
        role: "admin",
        createdAt: new Date().toISOString(),
      });
      this.saveUsers(users);
    }
  },
};

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

document.addEventListener("DOMContentLoaded", () => {
  Auth.seedAdmin();
});

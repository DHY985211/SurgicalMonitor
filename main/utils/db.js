/**
 * 本地简易数据库（实际项目可替换为SQLite或其他数据库）
 */
let sessions = [];
let sessionIdCounter = 1;

module.exports = {
    // 创建会话
    createSession: (sessionData) => {
        const session = {
            id: `session_${sessionIdCounter++}`,
            ...sessionData
        };
        sessions.push(session);
        return session.id;
    },

    // 获取所有会话
    getSessions: () => {
        return sessions;
    },

    // 根据ID获取会话
    getSessionById: (id) => {
        return sessions.find(s => s.id === id);
    },

    // 更新会话状态
    updateSessionStatus: (id, status) => {
        const session = sessions.find(s => s.id === id);
        if (session) {
            session.status = status;
            session.endTime = new Date().toISOString();
            return true;
        }
        return false;
    }
};
import * as Google from 'expo-auth-session/providers/google';
import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { GOOGLE_WEB_CLIENT_ID, GOOGLE_ANDROID_CLIENT_ID } from '../config/keys';

WebBrowser.maybeCompleteAuthSession();

// ─── Google OAuth Client IDs ─────────────────────────────────────────────────
const WEB_CLIENT_ID = GOOGLE_WEB_CLIENT_ID;
const ANDROID_CLIENT_ID = GOOGLE_ANDROID_CLIENT_ID;

// ─── Google Auth Hook ────────────────────────────────────────────────────────
// Returns the auth request, response, and promptAsync function.
// Uses expo-auth-session's Google provider for a clean OAuth flow.
export const useGoogleAuth = () => {
    // Check if we are running in Expo Go or a standalone build
    const isExpoGo = Constants.appOwnership === 'expo';

    console.log('[Auth] Environment:', isExpoGo ? 'Expo Go' : 'Standalone/Development Client');

    const [request, response, promptAsync] = Google.useAuthRequest({
        androidClientId: ANDROID_CLIENT_ID,
        webClientId: WEB_CLIENT_ID,
        scopes: [
            'openid',
            'profile',
            'email',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.file',
        ],
        // Let Expo handle the redirect URI automatically for better compatibility
    });

    return { request, response, promptAsync };
};

// ─── Get User Info from Google ───────────────────────────────────────────────
export const getUserInfo = async (accessToken) => {
    try {
        const res = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
            headers: { Authorization: `Bearer ${accessToken}` },
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error?.message || 'Failed to fetch user info');
        }

        return await res.json();
    } catch (err) {
        console.error('[Auth] getUserInfo failed:', err);
        return null;
    }
};

// ─── Token Exchange ──────────────────────────────────────────────────────────
// Exchanges an authorization code for an access token.
// Used when the Google provider returns a code instead of a token directly.
export const exchangeCodeForToken = async (code, codeVerifier, redirectUri) => {
    const tokenEndpoint = 'https://oauth2.googleapis.com/token';

    const params = {
        client_id: WEB_CLIENT_ID,
        code,
        code_verifier: codeVerifier,
        grant_type: 'authorization_code',
        redirect_uri: redirectUri,
    };

    console.log('[Auth] Exchanging code for token...');

    try {
        const res = await fetch(tokenEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(params).toString(),
        });

        const data = await res.json();

        if (data.error) {
            console.error('[Auth] Token exchange error:', data.error_description || data.error);
            throw new Error(data.error_description || data.error);
        }

        console.log('[Auth] Token exchange successful');
        return data;
    } catch (err) {
        console.error('[Auth] Token exchange failed:', err);
        throw err;
    }
};

// ─── Auth Data Persistence ───────────────────────────────────────────────────
const STORAGE_KEYS = {
    TOKEN: '@drivedrop_auth_token',
    USER: '@drivedrop_auth_user',
    EXPIRY: '@drivedrop_auth_expiry',
};

export const saveAuthData = async (token, user, expiresIn) => {
    const expiry = Date.now() + (expiresIn || 3600) * 1000;
    await AsyncStorage.multiSet([
        [STORAGE_KEYS.TOKEN, token],
        [STORAGE_KEYS.USER, JSON.stringify(user)],
        [STORAGE_KEYS.EXPIRY, expiry.toString()],
    ]);
};

export const getStoredAuthData = async () => {
    const results = await AsyncStorage.multiGet([
        STORAGE_KEYS.TOKEN,
        STORAGE_KEYS.USER,
        STORAGE_KEYS.EXPIRY,
    ]);

    const token = results[0][1];
    const user = results[1][1] ? JSON.parse(results[1][1]) : null;
    const expiry = results[2][1] ? parseInt(results[2][1], 10) : 0;

    if (!token || Date.now() > expiry) return null;
    return { accessToken: token, userInfo: user };
};

export const clearAuthData = async () => {
    await AsyncStorage.multiRemove([
        STORAGE_KEYS.TOKEN,
        STORAGE_KEYS.USER,
        STORAGE_KEYS.EXPIRY,
    ]);
};
